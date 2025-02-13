using UnityEngine;

public class SPH : MonoBehaviour
{
    [Header("Particle Settings")]
    [Tooltip("Mass of each particle. Derived from desired spacing and rest density.")]
    public float particleMass = 0.03f;  // Lower mass helps keep densities in check.
    [Tooltip("Rest (target) density of the fluid.")]
    public float restDensity = 1000f;
    // The Tait equation parameters:
    [Tooltip("Speed of sound in the fluid (affects compressibility).")]
    public float soundSpeed = 20f;
    [Tooltip("Gamma exponent for the Tait equation (typically ~7).")]
    public float gamma = 7f;
    [Tooltip("Stiffness factor is no longer used; pressure is computed using the Tait equation.")]
    public float stiffness = 800f;  // (Not used now; kept for legacy/debug purposes)
    [Tooltip("Viscosity coefficient.")]
    public float viscosity = 0.1f;
    [Tooltip("Smoothing (kernel) radius used for neighbor interactions. Should be ~1.3-2x the particle spacing.")]
    public float smoothingRadius = 0.2f;
    [Tooltip("Gravity constant (negative for downward).")]
    public float gravity = -9.81f;

    [Header("Time Settings")]
    [Tooltip("Base time step for the simulation.")]
    public float timeStep = 0.003f;
    [Tooltip("Number of solver sub-steps per frame.")]
    public int subSteps = 2;

    [Header("Surface Tension")]
    [Tooltip("Coefficient for surface tension forces.")]
    public float surfaceTensionCoefficient = 0.03f;

    [Header("XSPH Settings")]
    [Tooltip("Epsilon for XSPH velocity smoothing.")]
    public float xsphEpsilon = 0.5f;

    [Header("Axis-Aligned Boundary (Fallback)")]
    [Tooltip("Center of the simulation bounding box (used if no boundary cube is assigned).")]
    public Vector3 boundsCenter = Vector3.zero;
    [Tooltip("Size of the simulation bounding box (used if no boundary cube is assigned).")]
    public Vector3 boundsSize = new Vector3(10f, 10f, 10f);

    [Header("Boundary Cube")]
    [Tooltip("Assign a 3D object (e.g., a cube) whose edges define the simulation boundary.")]
    public Transform boundaryCube;

    [Header("Obstacle Settings")]
    [Tooltip("Optional sphere collider inside the fluid domain.")]
    public Collider obstacleCollider;
    [Tooltip("Repulsion stiffness for obstacle collisions.")]
    public float obstacleRepulsionStiffness = 5000f;
    [Tooltip("Velocity damping factor when colliding with obstacles.")]
    public float particleCollisionDamping = 0.9998f;

    [Header("Rendering Settings")]
    [Tooltip("Material for rendering the fluid (uses the GPU particle shader).")]
    public Material fluidMaterial;

    [Header("Compute Shaders")]
    [Tooltip("Main SPH compute (all kernels)")]
    public ComputeShader sphCompute;

    [Header("Grid Settings")]
    [Tooltip("Maximum number of particles allowed per grid cell.")]
    public int maxParticlesPerCell = 100;

    [Header("Spawn Boxes")]
    [Tooltip("List of SpawnBox objects that define where to spawn particles.")]
    public SpawnBox[] spawnBoxes;

    // ---------------------------
    // Internal Data Structures
    // ---------------------------
    // This structure must match the one in the compute shader.
    struct Particle
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 acceleration;
        public float density;
        public float pressure;
    }

    private ComputeBuffer particleBuffer;
    private ComputeBuffer gridCountsBuffer;
    private ComputeBuffer gridIndicesBuffer;

    // Grid parameters.
    private int gridResolutionX, gridResolutionY, gridResolutionZ;
    private int totalCells;
    private Vector3 gridMin;

    // Kernel indices.
    private int kernel_Clear;
    private int kernel_ClearGrid;
    private int kernel_BuildGrid;
    private int kernel_DensityPressure;
    private int kernel_ComputeForces;
    private int kernel_XSPH;
    private int kernel_VvHalfStep;
    private int kernel_VvFullStep;
    private int kernel_Boundaries;
    private int kernel_ObstacleCollision;
    private int kernel_VelocityDamping;

    private const int THREAD_GROUP_SIZE = 256;
    private int particleCount;

    void Start()
    {
        Application.targetFrameRate = 60;

        if (spawnBoxes == null || spawnBoxes.Length == 0)
        {
            spawnBoxes = FindObjectsOfType<SpawnBox>();
        }

        particleCount = 0;
        foreach (var sb in spawnBoxes)
        {
            particleCount += sb.particleCount;
        }

        CreateParticles();

        Vector3 halfBounds = boundsSize * 0.5f;
        gridMin = boundsCenter - halfBounds;
        gridResolutionX = Mathf.CeilToInt(boundsSize.x / smoothingRadius);
        gridResolutionY = Mathf.CeilToInt(boundsSize.y / smoothingRadius);
        gridResolutionZ = Mathf.CeilToInt(boundsSize.z / smoothingRadius);
        totalCells = gridResolutionX * gridResolutionY * gridResolutionZ;

        gridCountsBuffer = new ComputeBuffer(totalCells, sizeof(int));
        gridIndicesBuffer = new ComputeBuffer(totalCells * maxParticlesPerCell, sizeof(int));

        GetKernelIDs();
        SetComputeParams();
        BindBuffers();
        if (fluidMaterial != null)
        {
            fluidMaterial.SetBuffer("_ParticleBuffer", particleBuffer);
            fluidMaterial.SetInt("_NumParticles", particleCount);
        }
    }

    void Update()
    {
        // Update grid parameters from boundary cube’s AABB.
        if (boundaryCube != null)
        {
            Vector3[] corners = new Vector3[8];
            corners[0] = boundaryCube.TransformPoint(new Vector3(-0.5f, -0.5f, -0.5f));
            corners[1] = boundaryCube.TransformPoint(new Vector3(0.5f, -0.5f, -0.5f));
            corners[2] = boundaryCube.TransformPoint(new Vector3(0.5f, -0.5f, 0.5f));
            corners[3] = boundaryCube.TransformPoint(new Vector3(-0.5f, -0.5f, 0.5f));
            corners[4] = boundaryCube.TransformPoint(new Vector3(-0.5f, 0.5f, -0.5f));
            corners[5] = boundaryCube.TransformPoint(new Vector3(0.5f, 0.5f, -0.5f));
            corners[6] = boundaryCube.TransformPoint(new Vector3(0.5f, 0.5f, 0.5f));
            corners[7] = boundaryCube.TransformPoint(new Vector3(-0.5f, 0.5f, 0.5f));

            Vector3 newMin = corners[0], newMax = corners[0];
            for (int i = 1; i < 8; i++)
            {
                newMin = Vector3.Min(newMin, corners[i]);
                newMax = Vector3.Max(newMax, corners[i]);
            }

            Vector3 newBoundsSize = newMax - newMin;
            Vector3 newBoundsCenter = (newMin + newMax) * 0.5f;
            gridMin = newMin;

            int newGridResolutionX = Mathf.CeilToInt(newBoundsSize.x / smoothingRadius);
            int newGridResolutionY = Mathf.CeilToInt(newBoundsSize.y / smoothingRadius);
            int newGridResolutionZ = Mathf.CeilToInt(newBoundsSize.z / smoothingRadius);
            int newTotalCells = newGridResolutionX * newGridResolutionY * newGridResolutionZ;

            if (newTotalCells != totalCells)
            {
                gridCountsBuffer.Release();
                gridIndicesBuffer.Release();
                gridCountsBuffer = new ComputeBuffer(newTotalCells, sizeof(int));
                gridIndicesBuffer = new ComputeBuffer(newTotalCells * maxParticlesPerCell, sizeof(int));
                totalCells = newTotalCells;
            }

            gridResolutionX = newGridResolutionX;
            gridResolutionY = newGridResolutionY;
            gridResolutionZ = newGridResolutionZ;

            sphCompute.SetInt("_GridResolutionX", gridResolutionX);
            sphCompute.SetInt("_GridResolutionY", gridResolutionY);
            sphCompute.SetInt("_GridResolutionZ", gridResolutionZ);
            sphCompute.SetVector("_MinBound", gridMin);

            boundsCenter = newBoundsCenter;
            boundsSize = newBoundsSize;
        }

        UpdateObstacle();
        SetComputeParams();
        BindBuffers();

        DispatchCompute(kernel_ClearGrid, totalCells);
        DispatchCompute(kernel_BuildGrid, particleCount);

        float dtSub = timeStep / subSteps;
        sphCompute.SetFloat("_DeltaTime", dtSub);

        for (int i = 0; i < subSteps; i++)
        {
            DispatchCompute(kernel_VvHalfStep, particleCount);
            DispatchCompute(kernel_Clear, particleCount);
            DispatchCompute(kernel_DensityPressure, particleCount);
            DispatchCompute(kernel_ComputeForces, particleCount);
            DispatchCompute(kernel_XSPH, particleCount);
            DispatchCompute(kernel_VvFullStep, particleCount);
            DispatchCompute(kernel_Boundaries, particleCount);
            DispatchCompute(kernel_ObstacleCollision, particleCount);
        }
        DispatchCompute(kernel_VelocityDamping, particleCount);
    }

    void OnRenderObject()
    {
        if (fluidMaterial != null)
        {
            fluidMaterial.SetPass(0);
            Graphics.DrawProceduralNow(MeshTopology.Points, particleCount);
        }
    }

    void OnDestroy()
    {
        particleBuffer?.Release();
        gridCountsBuffer?.Release();
        gridIndicesBuffer?.Release();
    }

    void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
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

    // ----------------------------
    // Setup & Initialization Methods
    // ----------------------------
    private void CreateParticles()
    {
        int stride = sizeof(float) * (3 + 3 + 3 + 1 + 1);
        particleBuffer = new ComputeBuffer(particleCount, stride);

        Particle[] particlesInit = new Particle[particleCount];
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
                p.velocity = Vector3.zero;
                p.acceleration = Vector3.zero;
                p.density = restDensity;
                p.pressure = 0f;
                particlesInit[index++] = p;
            }
        }
        particleBuffer.SetData(particlesInit);
    }

    private void GetKernelIDs()
    {
        kernel_Clear = sphCompute.FindKernel("CS_Clear");
        kernel_ClearGrid = sphCompute.FindKernel("CS_ClearGrid");
        kernel_BuildGrid = sphCompute.FindKernel("CS_BuildGrid");
        kernel_DensityPressure = sphCompute.FindKernel("CS_DensityPressure");
        kernel_ComputeForces = sphCompute.FindKernel("CS_ComputeForces");
        kernel_XSPH = sphCompute.FindKernel("CS_XSPH");
        kernel_VvHalfStep = sphCompute.FindKernel("CS_VV_HalfStep");
        kernel_VvFullStep = sphCompute.FindKernel("CS_VV_FullStep");
        kernel_Boundaries = sphCompute.FindKernel("CS_Boundaries");
        kernel_ObstacleCollision = sphCompute.FindKernel("CS_ObstacleCollision");
        kernel_VelocityDamping = sphCompute.FindKernel("CS_VelocityDamping");

        Debug.Log("Kernel CS_Clear: " + kernel_Clear);
        Debug.Log("Kernel CS_ClearGrid: " + kernel_ClearGrid);
        Debug.Log("Kernel CS_BuildGrid: " + kernel_BuildGrid);
        Debug.Log("Kernel CS_DensityPressure: " + kernel_DensityPressure);
        Debug.Log("Kernel CS_ComputeForces: " + kernel_ComputeForces);
        Debug.Log("Kernel CS_XSPH: " + kernel_XSPH);
        Debug.Log("Kernel CS_VV_HalfStep: " + kernel_VvHalfStep);
        Debug.Log("Kernel CS_VV_FullStep: " + kernel_VvFullStep);
        Debug.Log("Kernel CS_Boundaries: " + kernel_Boundaries);
        Debug.Log("Kernel CS_ObstacleCollision: " + kernel_ObstacleCollision);
        Debug.Log("Kernel CS_VelocityDamping: " + kernel_VelocityDamping);
    }

    private void SetComputeParams()
    {
        sphCompute.SetInt("_ParticleCount", particleCount);
        sphCompute.SetFloat("_ParticleMass", particleMass);
        sphCompute.SetFloat("_RestDensity", restDensity);
        // Calculate the pressure constant B using the Tait equation:
        float B = restDensity * soundSpeed * soundSpeed / gamma;
        sphCompute.SetFloat("_B", B);
        sphCompute.SetFloat("_Gamma", gamma);
        // (Note: _Stiffness is no longer used for pressure calculation.)
        sphCompute.SetFloat("_Viscosity", viscosity);
        sphCompute.SetFloat("_SmoothingRadius", smoothingRadius);
        sphCompute.SetFloat("_Gravity", gravity);
        sphCompute.SetFloat("_SurfaceTensionCoefficient", surfaceTensionCoefficient);
        sphCompute.SetFloat("_XSPHEpsilon", xsphEpsilon);
        sphCompute.SetFloat("_ParticleCollisionDamping", particleCollisionDamping);
        sphCompute.SetFloat("_ObstacleRepulsionStiffness", obstacleRepulsionStiffness);

        sphCompute.SetInt("_GridResolutionX", gridResolutionX);
        sphCompute.SetInt("_GridResolutionY", gridResolutionY);
        sphCompute.SetInt("_GridResolutionZ", gridResolutionZ);
        sphCompute.SetInt("_MaxParticlesPerCell", maxParticlesPerCell);
        sphCompute.SetFloat("_CellSize", smoothingRadius);
        sphCompute.SetVector("_MinBound", gridMin);

        UpdateObstacle();

        if (boundaryCube != null)
        {
            Matrix4x4 boundaryMatrix = boundaryCube.localToWorldMatrix;
            Matrix4x4 boundaryInvMatrix = boundaryCube.worldToLocalMatrix;
            sphCompute.SetMatrix("_BoundaryMatrix", boundaryMatrix);
            sphCompute.SetMatrix("_BoundaryInvMatrix", boundaryInvMatrix);
            Vector3 halfExtents = new Vector3(0.5f, 0.5f, 0.5f);
            sphCompute.SetVector("_BoundaryHalfExtents", halfExtents);
        }
        else
        {
            sphCompute.SetVector("_BoundsCenter", boundsCenter);
            sphCompute.SetVector("_BoundsSize", boundsSize);
        }
    }

    private void UpdateObstacle()
    {
        if (obstacleCollider is SphereCollider sphere)
        {
            Vector3 pos = obstacleCollider.transform.position;
            float radius = sphere.radius * obstacleCollider.transform.lossyScale.x;
            sphCompute.SetVector("_ObstaclePos", pos);
            sphCompute.SetFloat("_ObstacleRadius", radius);
        }
        else
        {
            sphCompute.SetVector("_ObstaclePos", Vector3.zero);
            sphCompute.SetFloat("_ObstacleRadius", 0f);
        }
    }

    private void BindBuffers()
    {
        sphCompute.SetBuffer(kernel_Clear, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_BuildGrid, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_ComputeForces, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_XSPH, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VvHalfStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VvFullStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_Boundaries, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_ObstacleCollision, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VelocityDamping, "Particles", particleBuffer);

        sphCompute.SetBuffer(kernel_ClearGrid, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_BuildGrid, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_BuildGrid, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_ComputeForces, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_ComputeForces, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_XSPH, "GridCounts", gridCountsBuffer);
        sphCompute.SetBuffer(kernel_XSPH, "GridIndices", gridIndicesBuffer);
    }

    private void DispatchCompute(int kernelID, int count)
    {
        if (kernelID < 0)
        {
            Debug.LogError("Compute Shader kernel with ID " + kernelID + " is invalid!");
            return;
        }
        int groups = Mathf.CeilToInt(count / (float)THREAD_GROUP_SIZE);
        sphCompute.Dispatch(kernelID, groups, 1, 1);
    }

    public ComputeBuffer GetParticleBuffer() => particleBuffer;
    public int ParticleCount => particleCount;
}

