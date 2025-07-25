﻿using UnityEngine;

public class SPH : MonoBehaviour
{
    [Header("Particle Settings")]
    public float soundSpeed = 20f;
    public float gamma = 7f;
    public float smoothingRadius = 0.2f;  // Used in SPH physics but not for grid sizing.
    public float gravity = -9.81f;

    [Header("Time Settings")]
    public float timeStep = 0.003f;
    public int subSteps = 2;

    [Header("Surface Tension")]
    public float surfaceTensionCoefficient = 0.03f;
    // Only apply surface tension when the color field gradient exceeds this value.
    public float surfaceTensionThreshold = 0.01f;

    [Header("XSPH Settings")]
    public float xsphEpsilon = 0.5f;

    [Header("Axis-Aligned Boundary (Fallback)")]
    public Vector3 boundsCenter = Vector3.zero;
    public Vector3 boundsSize = new Vector3(10f, 10f, 10f);

    [Header("Boundary Cube")]
    public Transform boundaryCube;

    [Header("Obstacle Settings")]
    public Collider obstacleCollider;
    public float obstacleRepulsionStiffness = 5000f;
    public float particleCollisionDamping = 0.9999f;

    [Header("Rendering Settings")]
    public Material fluidMaterial;

    [Header("Compute Shaders")]
    public ComputeShader sphCompute;

    [Header("Grid Settings")]
    public int maxParticlesPerCell = 100;

    [Header("Grid Resolution")]
    [Tooltip("Number of cells along the smallest axis of the boundary. The grid will expand accordingly to cover the entire boundary.")]
    public int gridResolution = 50;

    [Header("Spawn Boxes")]
    public SpawnBox[] spawnBoxes;

    // ----------------------------------------------------------------
    // Particle layout (must match the compute kernels)
    // ----------------------------------------------------------------
    struct Particle
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 acceleration;
        public float density;
        public float pressure;
        public float restDensity;
        public float viscosity;
        public float mass;
        public Color color;
    }

    // GPU buffers
    private ComputeBuffer particleBuffer;
    private ComputeBuffer gridCountsBuffer;
    private ComputeBuffer gridIndicesBuffer;
    private ComputeBuffer collisionCounterBuffer;

    // CPU arrays
    private Particle[] particleArray;
    private int[] particleSpawnBoxIndices;

    // Grid parameters
    private int gridResolutionX, gridResolutionY, gridResolutionZ;
    public int totalCells; // Public for potential external use.
    private Vector3 gridMin;
    private float cellSize;  // Computed based on boundary size and gridResolution

    // Kernel IDs
    private int kernel_Clear;
    private int kernel_ClearGrid;
    private int kernel_BuildGrid;
    private int kernel_DensityPressure;
    private int kernel_ForceXSPH;
    private int kernel_VvHalfStep;
    private int kernel_VvFullStep;
    private int kernel_BoundObs;
    private int kernel_VelocityDamping;

    private const int THREAD_GROUP_SIZE = 256;
    private int particleCount;

    // For updating grid parameters when boundaryCube changes
    private Vector3 lastBoundaryCubePosition;
    private Quaternion lastBoundaryCubeRotation;
    private const float boundaryUpdateThreshold = 0.001f;

    // -------------------------------------------------------
    // Expose total particles and the particle buffer
    // -------------------------------------------------------
    public int ParticleCount => particleCount;

    public ComputeBuffer GetParticleBuffer()
    {
        return particleBuffer;
    }

    void Start()
    {
        // Set the target frame rate.
        Application.targetFrameRate = 60;

        // Find all spawn boxes if none are assigned.
        if (spawnBoxes == null || spawnBoxes.Length == 0)
        {
            spawnBoxes = FindObjectsOfType<SpawnBox>();
        }

        // Count total particles.
        particleCount = 0;
        foreach (var sb in spawnBoxes)
        {
            particleCount += sb.particleCount;
        }

        particleArray = new Particle[particleCount];
        particleSpawnBoxIndices = new int[particleCount];

        // Create particles and initialize the particle buffer.
        CreateParticles();

        // Setup the grid parameters based on the boundaryCube or fallback bounds.
        if (boundaryCube == null)
        {
            Vector3 half = boundsSize * 0.5f;
            gridMin = boundsCenter - half;
            Vector3 gridSize = boundsSize;
            float minSize = Mathf.Min(gridSize.x, gridSize.y, gridSize.z);
            cellSize = minSize / gridResolution;
            gridResolutionX = Mathf.CeilToInt(gridSize.x / cellSize);
            gridResolutionY = Mathf.CeilToInt(gridSize.y / cellSize);
            gridResolutionZ = Mathf.CeilToInt(gridSize.z / cellSize);
            totalCells = gridResolutionX * gridResolutionY * gridResolutionZ;
        }
        else
        {
            UpdateGridParametersFromBoundary();
            lastBoundaryCubePosition = boundaryCube.position;
            lastBoundaryCubeRotation = boundaryCube.rotation;
        }

        gridCountsBuffer = new ComputeBuffer(totalCells, sizeof(int));
        gridIndicesBuffer = new ComputeBuffer(totalCells * maxParticlesPerCell, sizeof(int));
        collisionCounterBuffer = new ComputeBuffer(1, sizeof(int));

        // Setup kernel IDs, compute parameters, and bind buffers.
        GetKernelIDs();
        SetComputeParams();
        BindBuffers();

        // Set the particle buffer in the fluid material if applicable.
        if (fluidMaterial != null)
        {
            fluidMaterial.SetBuffer("_ParticleBuffer", particleBuffer);
            fluidMaterial.SetInt("_NumParticles", particleCount);
        }
    }

    void Update()
    {
        // Update grid parameters if the boundaryCube has moved or rotated significantly.
        if (boundaryCube != null)
        {
            if (Vector3.Distance(boundaryCube.position, lastBoundaryCubePosition) > boundaryUpdateThreshold ||
                Quaternion.Angle(boundaryCube.rotation, lastBoundaryCubeRotation) > boundaryUpdateThreshold)
            {
                UpdateGridParametersFromBoundary();
                lastBoundaryCubePosition = boundaryCube.position;
                lastBoundaryCubeRotation = boundaryCube.rotation;
            }
        }

        UpdateObstacle();

        // Reset collision counter for this frame.
        collisionCounterBuffer.SetData(new int[] { 0 });

        // Update compute shader parameters and bind buffers.
        SetComputeParams();
        BindBuffers();

        // Update static properties (density, color, etc.) occasionally.
        // This can be adjusted based on performance requirements.
        UpdateParticleStaticProperties();

        // 1) Clear grid.
        DispatchCompute(kernel_ClearGrid, totalCells);

        // 2) Build grid.
        DispatchCompute(kernel_BuildGrid, particleCount);

        // Sub-step integration.
        float dtSub = timeStep / subSteps;
        sphCompute.SetFloat("_DeltaTime", dtSub);

        for (int i = 0; i < subSteps; i++)
        {
            DispatchCompute(kernel_VvHalfStep, particleCount);
            DispatchCompute(kernel_Clear, particleCount);
            DispatchCompute(kernel_DensityPressure, particleCount);
            DispatchCompute(kernel_ForceXSPH, particleCount);
            DispatchCompute(kernel_VvFullStep, particleCount);
            // Handle boundaries and obstacles.
            DispatchCompute(kernel_BoundObs, particleCount);
        }

        // Apply velocity damping.
        DispatchCompute(kernel_VelocityDamping, particleCount);
    }

    void OnRenderObject()
    {
        // Draw particles as points using the assigned fluid material.
        if (fluidMaterial != null)
        {
            fluidMaterial.SetPass(1);
            Graphics.DrawProceduralNow(MeshTopology.Points, particleCount);
        }
    }

    void OnDestroy()
    {
        particleBuffer?.Release();
        gridCountsBuffer?.Release();
        gridIndicesBuffer?.Release();
        collisionCounterBuffer?.Release();
    }

    // -------------------------------------------------------
    // Particle creation and initialization
    // -------------------------------------------------------
    private void CreateParticles()
    {
        int stride = sizeof(float) * 18;
        particleBuffer = new ComputeBuffer(particleCount, stride);

        Particle[] initArray = new Particle[particleCount];
        int index = 0;
        foreach (var sb in spawnBoxes)
        {
            Vector3 halfSpawn = sb.spawnSize * 0.5f;
            for (int i = 0; i < sb.particleCount; i++)
            {
                Vector3 randPos = new Vector3(
                    sb.SpawnCenter.x + Random.Range(-halfSpawn.x, halfSpawn.x),
                    sb.SpawnCenter.y + Random.Range(-halfSpawn.y, halfSpawn.y),
                    sb.SpawnCenter.z + Random.Range(-halfSpawn.z, halfSpawn.z)
                );

                Particle p;
                p.position = randPos;
                p.velocity = sb.initialVelocity;
                p.acceleration = Vector3.zero;
                p.density = sb.restDensity;
                p.pressure = 0f;
                p.restDensity = sb.restDensity;
                p.viscosity = sb.viscosity;
                p.mass = sb.particleMass;
                p.color = sb.particleColor;

                initArray[index] = p;
                particleSpawnBoxIndices[index] = System.Array.IndexOf(spawnBoxes, sb);
                index++;
            }
        }
        particleBuffer.SetData(initArray);
        System.Array.Copy(initArray, particleArray, particleCount);
    }

    private void UpdateParticleStaticProperties()
    {
        // Read from GPU.
        particleBuffer.GetData(particleArray);
        for (int i = 0; i < particleCount; i++)
        {
            SpawnBox sb = spawnBoxes[particleSpawnBoxIndices[i]];
            particleArray[i].restDensity = sb.restDensity;
            particleArray[i].viscosity = sb.viscosity;
            particleArray[i].mass = sb.particleMass;
            particleArray[i].color = sb.particleColor;
        }
        // Update GPU with the new static properties.
        particleBuffer.SetData(particleArray);
    }

    // -------------------------------------------------------
    // Grid / Boundary Setup
    // -------------------------------------------------------
    private void UpdateGridParametersFromBoundary()
    {
        if (boundaryCube == null)
            return;

        // Get the eight corners of the boundary cube.
        Vector3[] corners = new Vector3[8];
        corners[0] = boundaryCube.TransformPoint(new Vector3(-0.5f, -0.5f, -0.5f));
        corners[1] = boundaryCube.TransformPoint(new Vector3(0.5f, -0.5f, -0.5f));
        corners[2] = boundaryCube.TransformPoint(new Vector3(0.5f, -0.5f, 0.5f));
        corners[3] = boundaryCube.TransformPoint(new Vector3(-0.5f, -0.5f, 0.5f));
        corners[4] = boundaryCube.TransformPoint(new Vector3(-0.5f, 0.5f, -0.5f));
        corners[5] = boundaryCube.TransformPoint(new Vector3(0.5f, 0.5f, -0.5f));
        corners[6] = boundaryCube.TransformPoint(new Vector3(0.5f, 0.5f, 0.5f));
        corners[7] = boundaryCube.TransformPoint(new Vector3(-0.5f, 0.5f, 0.5f));

        Vector3 newMin = corners[0];
        Vector3 newMax = corners[0];
        for (int i = 1; i < 8; i++)
        {
            newMin = Vector3.Min(newMin, corners[i]);
            newMax = Vector3.Max(newMax, corners[i]);
        }
        Vector3 gridSize = newMax - newMin;
        gridMin = newMin;

        // Use the user-defined gridResolution.
        int effectiveGridRes = gridResolution;
        float minSize = Mathf.Min(gridSize.x, gridSize.y, gridSize.z);
        cellSize = minSize / effectiveGridRes;
        gridResolutionX = Mathf.CeilToInt(gridSize.x / cellSize);
        gridResolutionY = Mathf.CeilToInt(gridSize.y / cellSize);
        gridResolutionZ = Mathf.CeilToInt(gridSize.z / cellSize);
        totalCells = gridResolutionX * gridResolutionY * gridResolutionZ;

        // Reallocate grid buffers with the new total cell count.
        if (gridCountsBuffer != null)
        {
            gridCountsBuffer.Release();
            gridIndicesBuffer.Release();
        }
        gridCountsBuffer = new ComputeBuffer(totalCells, sizeof(int));
        gridIndicesBuffer = new ComputeBuffer(totalCells * maxParticlesPerCell, sizeof(int));
    }

    private void UpdateObstacle()
    {
        if (obstacleCollider is SphereCollider sphere)
        {
            float radius = sphere.radius * obstacleCollider.transform.lossyScale.x;
            sphCompute.SetVector("_ObstaclePos", obstacleCollider.transform.position);
            sphCompute.SetFloat("_ObstacleRadius", radius);
        }
        else
        {
            sphCompute.SetVector("_ObstaclePos", Vector3.zero);
            sphCompute.SetFloat("_ObstacleRadius", 0f);
        }
    }

    // -------------------------------------------------------
    // Kernel Setup and Dispatch
    // -------------------------------------------------------
    private void GetKernelIDs()
    {
        kernel_Clear = sphCompute.FindKernel("CS_Clear");
        kernel_ClearGrid = sphCompute.FindKernel("CS_ClearGrid");
        kernel_BuildGrid = sphCompute.FindKernel("CS_BuildGrid");
        kernel_DensityPressure = sphCompute.FindKernel("CS_DensityPressure");
        kernel_ForceXSPH = sphCompute.FindKernel("CS_ForceXSPH");
        kernel_VvHalfStep = sphCompute.FindKernel("CS_VvHalfStep");
        kernel_VvFullStep = sphCompute.FindKernel("CS_VvFullStep");
        kernel_BoundObs = sphCompute.FindKernel("CS_BoundObs");
        kernel_VelocityDamping = sphCompute.FindKernel("CS_VelocityDamping");
    }

    private void SetComputeParams()
    {
        sphCompute.SetInt("_ParticleCount", particleCount);
        sphCompute.SetFloat("_SoundSpeed", soundSpeed);
        sphCompute.SetFloat("_Gamma", gamma);
        sphCompute.SetFloat("_SmoothingRadius", smoothingRadius);
        sphCompute.SetFloat("_Gravity", gravity);
        sphCompute.SetFloat("_SurfaceTensionCoefficient", surfaceTensionCoefficient);
        sphCompute.SetFloat("_SurfaceTensionThreshold", surfaceTensionThreshold);
        sphCompute.SetFloat("_XSPHEpsilon", xsphEpsilon);
        sphCompute.SetFloat("_ParticleCollisionDamping", particleCollisionDamping);
        sphCompute.SetFloat("_ObstacleRepulsionStiffness", obstacleRepulsionStiffness);

        sphCompute.SetInt("_GridResolutionX", gridResolutionX);
        sphCompute.SetInt("_GridResolutionY", gridResolutionY);
        sphCompute.SetInt("_GridResolutionZ", gridResolutionZ);
        sphCompute.SetInt("_MaxParticlesPerCell", maxParticlesPerCell);
        sphCompute.SetFloat("_CellSize", cellSize);
        sphCompute.SetVector("_MinBound", gridMin);

        if (boundaryCube != null)
        {
            Matrix4x4 m = boundaryCube.localToWorldMatrix;
            Matrix4x4 invM = boundaryCube.worldToLocalMatrix;
            sphCompute.SetMatrix("_BoundaryMatrix", m);
            sphCompute.SetMatrix("_BoundaryInvMatrix", invM);
            sphCompute.SetVector("_BoundaryHalfExtents", new Vector3(0.5f, 0.5f, 0.5f));
        }
        else
        {
            sphCompute.SetVector("_BoundsCenter", boundsCenter);
            sphCompute.SetVector("_BoundsSize", boundsSize);
        }
    }

    private void BindBuffers()
    {
        // Bind the particle buffer.
        sphCompute.SetBuffer(kernel_Clear, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_BuildGrid, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_ForceXSPH, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VvHalfStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VvFullStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_BoundObs, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VelocityDamping, "Particles", particleBuffer);

        // Bind the grid buffers.
        sphCompute.SetBuffer(kernel_ClearGrid, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_BuildGrid, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_BuildGrid, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_ForceXSPH, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_ForceXSPH, "GridIndices", gridIndicesBuffer);

        // Bind the collision counter buffer for the boundary kernel.
        sphCompute.SetBuffer(kernel_BoundObs, "CollisionCounter", collisionCounterBuffer);
    }

    private void DispatchCompute(int kernel, int count)
    {
        int groups = Mathf.CeilToInt(count / (float)THREAD_GROUP_SIZE);
        sphCompute.Dispatch(kernel, groups, 1, 1);
    }
}
