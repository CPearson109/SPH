using UnityEngine;

public class SPH : MonoBehaviour
{
    [Header("Particle Settings")]
    // Removed global controls for particleMass, restDensity, and viscosity.
    // These are now controlled per-particle via SpawnBox.
    public float soundSpeed = 20f;
    public float gamma = 7f;
    public float smoothingRadius = 0.2f;
    public float gravity = -9.81f;

    [Header("Time Settings")]
    public float timeStep = 0.003f;
    public int subSteps = 2;

    [Header("Surface Tension")]
    public float surfaceTensionCoefficient = 0.03f;

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
    public float particleCollisionDamping = 0.9998f;

    [Header("Rendering Settings")]
    public Material fluidMaterial;

    [Header("Compute Shaders")]
    public ComputeShader sphCompute;

    [Header("Grid Settings")]
    public int maxParticlesPerCell = 100;

    [Header("Spawn Boxes")]
    public SpawnBox[] spawnBoxes;

    /*
      Optimized Fixed-Radius Nearest Neighbor Search (NNS):
      - The simulation space is partitioned into uniform grid cells.
      - Particles are inserted into grid cells using atomic operations.
      - Only nearby cells are processed during neighbor search.
      - This reduces computational complexity from O(n²) to roughly O(n*k).
    */

    // ----------------------------------------------------------------
    // Each particle stores 18 floats in total:
    //
    //    position (3) + velocity (3) + acceleration (3)
    //  + density (1) + pressure (1)
    //  + restDensity (1) + viscosity (1) + mass (1)
    //  + color (4)
    //
    // ----------------------------------------------------------------
    struct Particle
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 acceleration;
        public float density;
        public float pressure;
        public float restDensity; // from SpawnBox
        public float viscosity;   // from SpawnBox
        public float mass;        // from SpawnBox
        public Color color;       // from SpawnBox (for multiphase visualization)
    }

    private ComputeBuffer particleBuffer;
    private ComputeBuffer gridCountsBuffer;
    private ComputeBuffer gridIndicesBuffer;

    // CPU-side arrays for dynamic updates or inspection.
    private Particle[] particleArray;
    private int[] particleSpawnBoxIndices; // tracks which SpawnBox each particle came from

    // Grid parameters
    private int gridResolutionX, gridResolutionY, gridResolutionZ;
    private int totalCells;
    private Vector3 gridMin;

    // Kernel indices in the compute shader
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

    // For updating static properties less frequently:
    private float staticUpdateTimer = 0f;
    private const float staticUpdateInterval = 0.1f;

    // For detecting boundaryCube movement changes:
    private Vector3 lastBoundaryCubePosition;
    private Quaternion lastBoundaryCubeRotation;
    private const float boundaryUpdateThreshold = 0.001f;

    void Start()
    {
        // Limit frame rate for consistent testing
        Application.targetFrameRate = 60;

        // Collect SpawnBoxes if not assigned
        if (spawnBoxes == null || spawnBoxes.Length == 0)
        {
            spawnBoxes = FindObjectsOfType<SpawnBox>();
        }

        // Count total particles from all SpawnBoxes
        particleCount = 0;
        foreach (var sb in spawnBoxes)
        {
            particleCount += sb.particleCount;
        }

        // Allocate CPU-side arrays
        particleArray = new Particle[particleCount];
        particleSpawnBoxIndices = new int[particleCount];

        // Create and initialize the particles
        CreateParticles();

        // Setup initial grid parameters
        if (boundaryCube == null)
        {
            // Fallback to axis-aligned bounding box
            Vector3 halfBounds = boundsSize * 0.5f;
            gridMin = boundsCenter - halfBounds;
            gridResolutionX = Mathf.CeilToInt(boundsSize.x / smoothingRadius);
            gridResolutionY = Mathf.CeilToInt(boundsSize.y / smoothingRadius);
            gridResolutionZ = Mathf.CeilToInt(boundsSize.z / smoothingRadius);
            totalCells = gridResolutionX * gridResolutionY * gridResolutionZ;
        }
        else
        {
            // Cache the initial transform for boundaryCube
            lastBoundaryCubePosition = boundaryCube.position;
            lastBoundaryCubeRotation = boundaryCube.rotation;
            UpdateGridParametersFromBoundary();
        }

        // Allocate grid buffers
        gridCountsBuffer = new ComputeBuffer(totalCells, sizeof(int));
        gridIndicesBuffer = new ComputeBuffer(totalCells * maxParticlesPerCell, sizeof(int));

        // Get all kernel IDs and set initial parameters
        GetKernelIDs();
        SetComputeParams();
        BindBuffers();

        // Pass the particle buffer to the fluid material for rendering
        if (fluidMaterial != null)
        {
            fluidMaterial.SetBuffer("_ParticleBuffer", particleBuffer);
            fluidMaterial.SetInt("_NumParticles", particleCount);
        }
    }

    void Update()
    {
        // If boundaryCube has moved or rotated significantly, update the grid
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

        // Keep obstacle data fresh
        UpdateObstacle();

        // Re-send general parameters in case they changed
        SetComputeParams();
        BindBuffers();

        // Update static properties (restDensity, mass, color, etc.) less often
        staticUpdateTimer += Time.deltaTime;
        if (staticUpdateTimer >= staticUpdateInterval)
        {
            UpdateParticleStaticProperties();
            staticUpdateTimer = 0f;
        }

        // 1) Clear grid cell counts
        DispatchCompute(kernel_ClearGrid, totalCells);

        // 2) Re-build grid (assign each particle to a cell)
        DispatchCompute(kernel_BuildGrid, particleCount);

        // Sub-step the SPH solver
        float dtSub = timeStep / subSteps;
        sphCompute.SetFloat("_DeltaTime", dtSub);
        for (int i = 0; i < subSteps; i++)
        {
            // Velocity Verlet half-step
            DispatchCompute(kernel_VvHalfStep, particleCount);

            // Clear per-particle neighbor accumulators
            DispatchCompute(kernel_Clear, particleCount);

            // Density & Pressure
            DispatchCompute(kernel_DensityPressure, particleCount);

            // Forces + XSPH correction
            DispatchCompute(kernel_ForceXSPH, particleCount);

            // Velocity Verlet full-step
            DispatchCompute(kernel_VvFullStep, particleCount);

            // Boundary + Obstacle collision
            DispatchCompute(kernel_BoundObs, particleCount);
        }

        // Very slight damping to prevent perpetual bounce
        DispatchCompute(kernel_VelocityDamping, particleCount);
    }

    void OnRenderObject()
    {
        // Render the particles as points; the vertex/geometry shader must use the color
        if (fluidMaterial != null)
        {
            fluidMaterial.SetPass(1);
            Graphics.DrawProceduralNow(MeshTopology.Points, particleCount);
        }
    }

    void OnDestroy()
    {
        // Cleanup
        particleBuffer?.Release();
        gridCountsBuffer?.Release();
        gridIndicesBuffer?.Release();
    }

    void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        // Draw boundary cube if assigned, otherwise draw AABB
        if (boundaryCube != null)
        {
            Gizmos.matrix = boundaryCube.localToWorldMatrix;
            Gizmos.DrawWireCube(Vector3.zero, Vector3.one);
            Gizmos.matrix = Matrix4x4.identity;
        }
        else
        {
            Gizmos.DrawWireCube(boundsCenter, boundsSize);
        }
    }

    // ----------------------------------------------------------------
    // Particle Creation & Setup
    // ----------------------------------------------------------------

    private void CreateParticles()
    {
        // Each particle has 18 floats in the struct.
        int stride = sizeof(float) * 18;
        particleBuffer = new ComputeBuffer(particleCount, stride);

        Particle[] particlesInit = new Particle[particleCount];

        int index = 0;
        for (int sbIndex = 0; sbIndex < spawnBoxes.Length; sbIndex++)
        {
            SpawnBox sb = spawnBoxes[sbIndex];

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
                p.restDensity = sb.restDensity;
                p.viscosity = sb.viscosity;
                p.mass = sb.particleMass;
                p.color = sb.particleColor;
                // Initialize density & pressure
                p.density = p.restDensity;
                p.pressure = 0f;

                particlesInit[index] = p;
                particleSpawnBoxIndices[index] = sbIndex;
                index++;
            }
        }

        // Upload to the GPU buffer
        particleBuffer.SetData(particlesInit);
        // Keep a CPU copy for partial updates
        System.Array.Copy(particlesInit, particleArray, particleCount);
    }

    // Update static properties (restDensity, mass, color, etc.) from each SpawnBox
    private void UpdateParticleStaticProperties()
    {
        // We read back the entire array, update the static fields, then send back to GPU
        particleBuffer.GetData(particleArray);
        for (int i = 0; i < particleCount; i++)
        {
            int sbIndex = particleSpawnBoxIndices[i];
            SpawnBox sb = spawnBoxes[sbIndex];

            particleArray[i].restDensity = sb.restDensity;
            particleArray[i].viscosity = sb.viscosity;
            particleArray[i].mass = sb.particleMass;
            particleArray[i].color = sb.particleColor;
        }
        particleBuffer.SetData(particleArray);
    }

    // ----------------------------------------------------------------
    // Boundary & Grid Setup
    // ----------------------------------------------------------------

    private void UpdateGridParametersFromBoundary()
    {
        if (boundaryCube == null) return;

        // Get the world positions of the 8 corners of the boundary cube
        Vector3[] corners = new Vector3[8];
        corners[0] = boundaryCube.TransformPoint(new Vector3(-0.5f, -0.5f, -0.5f));
        corners[1] = boundaryCube.TransformPoint(new Vector3(0.5f, -0.5f, -0.5f));
        corners[2] = boundaryCube.TransformPoint(new Vector3(0.5f, -0.5f, 0.5f));
        corners[3] = boundaryCube.TransformPoint(new Vector3(-0.5f, -0.5f, 0.5f));
        corners[4] = boundaryCube.TransformPoint(new Vector3(-0.5f, 0.5f, -0.5f));
        corners[5] = boundaryCube.TransformPoint(new Vector3(0.5f, 0.5f, -0.5f));
        corners[6] = boundaryCube.TransformPoint(new Vector3(0.5f, 0.5f, 0.5f));
        corners[7] = boundaryCube.TransformPoint(new Vector3(-0.5f, 0.5f, 0.5f));

        // Compute new min/max in world space
        Vector3 newMin = corners[0];
        Vector3 newMax = corners[0];
        for (int i = 1; i < 8; i++)
        {
            newMin = Vector3.Min(newMin, corners[i]);
            newMax = Vector3.Max(newMax, corners[i]);
        }

        // Update
        Vector3 newBoundsSize = newMax - newMin;
        Vector3 newBoundsCenter = (newMin + newMax) * 0.5f;
        gridMin = newMin;

        int newGridX = Mathf.CeilToInt(newBoundsSize.x / smoothingRadius);
        int newGridY = Mathf.CeilToInt(newBoundsSize.y / smoothingRadius);
        int newGridZ = Mathf.CeilToInt(newBoundsSize.z / smoothingRadius);
        int newTotalCells = newGridX * newGridY * newGridZ;

        // If the total cell count changed, reallocate buffers
        if (newTotalCells != totalCells)
        {
            gridCountsBuffer?.Release();
            gridIndicesBuffer?.Release();

            gridCountsBuffer = new ComputeBuffer(newTotalCells, sizeof(int));
            gridIndicesBuffer = new ComputeBuffer(newTotalCells * maxParticlesPerCell, sizeof(int));

            totalCells = newTotalCells;
        }

        gridResolutionX = newGridX;
        gridResolutionY = newGridY;
        gridResolutionZ = newGridZ;

        // Also store for fallback Gizmos
        boundsCenter = newBoundsCenter;
        boundsSize = newBoundsSize;
    }

    private void UpdateObstacle()
    {
        // Example: if we have a sphere collider as an obstacle
        if (obstacleCollider is SphereCollider sphere)
        {
            Vector3 pos = obstacleCollider.transform.position;
            float radius = sphere.radius * obstacleCollider.transform.lossyScale.x;
            sphCompute.SetVector("_ObstaclePos", pos);
            sphCompute.SetFloat("_ObstacleRadius", radius);
        }
        else
        {
            // No obstacle or unsupported type
            sphCompute.SetVector("_ObstaclePos", Vector3.zero);
            sphCompute.SetFloat("_ObstacleRadius", 0f);
        }
    }

    // ----------------------------------------------------------------
    // Compute Shader Setup
    // ----------------------------------------------------------------
    private void GetKernelIDs()
    {
        kernel_Clear = sphCompute.FindKernel("CS_Clear");
        kernel_ClearGrid = sphCompute.FindKernel("CS_ClearGrid");
        kernel_BuildGrid = sphCompute.FindKernel("CS_BuildGrid");
        kernel_DensityPressure = sphCompute.FindKernel("CS_DensityPressure");
        kernel_ForceXSPH = sphCompute.FindKernel("CS_ForceXSPH");
        kernel_VvHalfStep = sphCompute.FindKernel("CS_VV_HalfStep");
        kernel_VvFullStep = sphCompute.FindKernel("CS_VV_FullStep");
        kernel_BoundObs = sphCompute.FindKernel("CS_BoundObs");
        kernel_VelocityDamping = sphCompute.FindKernel("CS_VelocityDamping");

        Debug.Log("Kernel IDs:");
        Debug.Log($" - CS_Clear: {kernel_Clear}");
        Debug.Log($" - CS_ClearGrid: {kernel_ClearGrid}");
        Debug.Log($" - CS_BuildGrid: {kernel_BuildGrid}");
        Debug.Log($" - CS_DensityPressure: {kernel_DensityPressure}");
        Debug.Log($" - CS_ForceXSPH: {kernel_ForceXSPH}");
        Debug.Log($" - CS_VV_HalfStep: {kernel_VvHalfStep}");
        Debug.Log($" - CS_VV_FullStep: {kernel_VvFullStep}");
        Debug.Log($" - CS_BoundObs: {kernel_BoundObs}");
        Debug.Log($" - CS_VelocityDamping: {kernel_VelocityDamping}");
    }

    private void SetComputeParams()
    {
        // General fluid parameters
        sphCompute.SetInt("_ParticleCount", particleCount);
        sphCompute.SetFloat("_SoundSpeed", soundSpeed);
        sphCompute.SetFloat("_Gamma", gamma);
        sphCompute.SetFloat("_SmoothingRadius", smoothingRadius);
        sphCompute.SetFloat("_Gravity", gravity);
        sphCompute.SetFloat("_SurfaceTensionCoefficient", surfaceTensionCoefficient);
        sphCompute.SetFloat("_XSPHEpsilon", xsphEpsilon);
        sphCompute.SetFloat("_ParticleCollisionDamping", particleCollisionDamping);
        sphCompute.SetFloat("_ObstacleRepulsionStiffness", obstacleRepulsionStiffness);

        // Grid parameters
        sphCompute.SetInt("_GridResolutionX", gridResolutionX);
        sphCompute.SetInt("_GridResolutionY", gridResolutionY);
        sphCompute.SetInt("_GridResolutionZ", gridResolutionZ);
        sphCompute.SetInt("_MaxParticlesPerCell", maxParticlesPerCell);
        sphCompute.SetFloat("_CellSize", smoothingRadius);
        sphCompute.SetVector("_MinBound", gridMin);

        // Obstacle
        UpdateObstacle();

        // Boundary transform or fallback bounding box
        if (boundaryCube != null)
        {
            Matrix4x4 boundaryMatrix = boundaryCube.localToWorldMatrix;
            Matrix4x4 boundaryInvMatrix = boundaryCube.worldToLocalMatrix;
            sphCompute.SetMatrix("_BoundaryMatrix", boundaryMatrix);
            sphCompute.SetMatrix("_BoundaryInvMatrix", boundaryInvMatrix);
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
        // Particle buffer binds
        sphCompute.SetBuffer(kernel_Clear, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_BuildGrid, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_ForceXSPH, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VvHalfStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VvFullStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_BoundObs, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VelocityDamping, "Particles", particleBuffer);

        // Grid buffers
        sphCompute.SetBuffer(kernel_ClearGrid, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_BuildGrid, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_BuildGrid, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_ForceXSPH, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_ForceXSPH, "GridIndices", gridIndicesBuffer);
    }

    private void DispatchCompute(int kernelID, int count)
    {
        if (kernelID < 0)
        {
            Debug.LogError($"Compute Shader kernel with ID {kernelID} is invalid!");
            return;
        }
        int groups = Mathf.CeilToInt(count / (float)THREAD_GROUP_SIZE);
        sphCompute.Dispatch(kernelID, groups, 1, 1);
    }

    // Optional accessors for other scripts or debugging
    public ComputeBuffer GetParticleBuffer() => particleBuffer;
    public int ParticleCount => particleCount;
}
