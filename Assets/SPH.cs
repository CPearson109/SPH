using UnityEngine;

public class SPH : MonoBehaviour
{
    [Header("Particle Settings")]
    [Tooltip("Total number of particles in the simulation.")]
    public int particleCount = 50000;

    [Tooltip("Mass of each particle.")]
    public float particleMass = 0.8f;

    [Tooltip("Rest (target) density of the fluid.")]
    public float restDensity = 1000f;

    [Tooltip("Stiffness factor (bulk modulus).")]
    public float stiffness = 5000f;

    [Tooltip("Viscosity coefficient.")]
    public float viscosity = 0.1f;

    [Tooltip("Smoothing (kernel) radius used for neighbor interactions.")]
    public float smoothingRadius = 0.3f;

    [Tooltip("Gravity constant (negative for downward).")]
    public float gravity = -9.81f;

    [Header("Spatial Hashing Settings")]
    [Tooltip("Resolution of the 3D grid used for spatial hashing (automatically set if autoAdjustSpatialHashing is true).")]
    public Vector3Int gridResolution = new Vector3Int(64, 64, 64);

    [Tooltip("Maximum particles per grid cell (automatically set if autoAdjustSpatialHashing is true).")]
    public int maxParticlesPerCell = 100;

    [Tooltip("Radius used to determine grid cell index (if zero or negative, it defaults to smoothingRadius).")]
    public float neighborSearchRadius = 0.3f;

    [Tooltip("Automatically adjust spatial hashing parameters based on boundary size and particle count.")]
    public bool autoAdjustSpatialHashing = true;

    [Header("Time Settings")]
    [Tooltip("Base time step for the simulation.")]
    public float timeStep = 0.003f;

    [Tooltip("Number of solver sub-steps per frame.")]
    public int subSteps = 2;

    [Header("Surface Tension")]
    [Tooltip("Coefficient for surface tension forces.")]
    public float surfaceTensionCoefficient = 0.1f;

    [Header("XSPH Settings")]
    [Tooltip("Epsilon for XSPH velocity smoothing.")]
    public float xsphEpsilon = 0.5f;

    [Header("Boundary Settings")]
    [Tooltip("Center of the simulation bounding box.")]
    public Vector3 boundsCenter = Vector3.zero;

    [Tooltip("Size of the simulation bounding box.")]
    public Vector3 boundsSize = new Vector3(10f, 10f, 10f);

    [Header("Spawn Box Settings")]
    [Tooltip("Center of the region where particles are initially spawned.")]
    public Vector3 spawnCenter = new Vector3(0, 2, 0);

    [Tooltip("Size of the region where particles are initially spawned.")]
    public Vector3 spawnSize = new Vector3(2f, 2f, 2f);

    [Header("Obstacle Settings")]
    [Tooltip("Optional sphere collider inside the fluid domain.")]
    public Collider obstacleCollider;

    [Tooltip("Repulsion stiffness for obstacle collisions.")]
    public float obstacleRepulsionStiffness = 5000f;

    [Tooltip("Velocity damping factor when colliding with obstacles.")]
    public float particleCollisionDamping = 0.98f;

    [Header("Rendering Settings")]
    [Tooltip("Material for rendering the fluid (uses the GPU particle shader).")]
    public Material fluidMaterial;

    [Header("Compute Shaders")]
    [Tooltip("Main SPH compute (all kernels)")]
    public ComputeShader sphCompute;

    // ----- Internal Buffers & Data Structures -----
    // Must match the struct in the compute shader.
    struct Particle
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 acceleration;
        public float density;
        public float pressure;
    }

    private ComputeBuffer particleBuffer;      // All particles
    private ComputeBuffer gridBuffer;            // Stores up to maxParticlesPerCell indices per cell
    private ComputeBuffer gridIndicesBuffer;     // Stores the count of particles in each cell

    private int gridCellCount;

    // Kernel indices
    private int kernel_Clear;
    private int kernel_ClearGrid;
    private int kernel_SpatialHash;
    private int kernel_DensityPressure;
    private int kernel_ComputeForces;
    private int kernel_XSPH;
    private int kernel_VvHalfStep;
    private int kernel_VvFullStep;
    private int kernel_Boundaries;
    private int kernel_ObstacleCollision;
    private int kernel_VelocityDamping;  // New kernel for additional velocity damping

    // For dispatching compute threads
    private const int THREAD_GROUP_SIZE = 256;

    void Start()
    {
        // Optionally limit frame rate for consistency
        Application.targetFrameRate = 60;

        // Automatically adjust spatial hashing parameters if enabled.
        if (autoAdjustSpatialHashing)
        {
            AdjustSpatialHashingParameters();
        }

        // 1) Initialize particles and GPU buffer
        CreateParticles();

        // 2) Initialize the grid buffers (using the adjusted gridResolution and maxParticlesPerCell)
        CreateGrid();

        // 3) Get compute shader kernels and set simulation parameters
        GetKernelIDs();
        SetComputeParams();

        // 4) Bind the buffers to all compute shader kernels
        BindBuffers();

        // Bind the particle buffer to the rendering material
        if (fluidMaterial != null)
        {
            fluidMaterial.SetBuffer("_ParticleBuffer", particleBuffer);
            fluidMaterial.SetInt("_NumParticles", particleCount);
        }
    }

    void Update()
    {
        // Update dynamic parameters (e.g. obstacle position)
        UpdateObstacle();
        SetComputeParams();

        // Set sub-step time
        float dtSub = timeStep / subSteps;
        sphCompute.SetFloat("_DeltaTime", dtSub);

        // 1) Clear the spatial grid
        DispatchCompute(kernel_ClearGrid, gridCellCount);

        // 2) Assign particles to grid cells via spatial hashing
        DispatchCompute(kernel_SpatialHash, particleCount);

        // 3) Run the multi-step solver
        for (int i = 0; i < subSteps; i++)
        {
            // 3a) Velocity Verlet half-step
            DispatchCompute(kernel_VvHalfStep, particleCount);

            // 3b) Clear per-particle accumulators (acceleration, density, pressure)
            DispatchCompute(kernel_Clear, particleCount);

            // 3c) Compute density and pressure for each particle
            DispatchCompute(kernel_DensityPressure, particleCount);

            // 3d) Compute forces (pressure, viscosity, surface tension)
            DispatchCompute(kernel_ComputeForces, particleCount);

            // 3e) XSPH velocity correction for smoothing
            DispatchCompute(kernel_XSPH, particleCount);

            // 3f) Velocity Verlet full-step update
            DispatchCompute(kernel_VvFullStep, particleCount);

            // 3g) Handle boundaries and obstacle collisions
            DispatchCompute(kernel_Boundaries, particleCount);
            DispatchCompute(kernel_ObstacleCollision, particleCount);
        }

        // 4) Additional velocity damping to help particles settle
        DispatchCompute(kernel_VelocityDamping, particleCount);
    }

    // Render the particles on the GPU
    void OnRenderObject()
    {
        if (fluidMaterial != null)
        {
            fluidMaterial.SetPass(0);
            // Draw all particles as points using a single draw call.
            Graphics.DrawProceduralNow(MeshTopology.Points, particleCount);
        }
    }

    private void OnDestroy()
    {
        // Release all GPU buffers
        particleBuffer?.Release();
        gridBuffer?.Release();
        gridIndicesBuffer?.Release();
    }

    // ------------------------------------------------------
    // Spatial Hashing Auto-Adjustment
    // ------------------------------------------------------
    private void AdjustSpatialHashingParameters()
    {
        // Ensure neighborSearchRadius is valid; if not, default it to the smoothing radius.
        float cellSize = (neighborSearchRadius > 0) ? neighborSearchRadius : smoothingRadius;

        // Compute grid resolution as the number of cells needed along each axis
        gridResolution = new Vector3Int(
            Mathf.CeilToInt(boundsSize.x / cellSize),
            Mathf.CeilToInt(boundsSize.y / cellSize),
            Mathf.CeilToInt(boundsSize.z / cellSize)
        );

        // Total number of cells
        gridCellCount = gridResolution.x * gridResolution.y * gridResolution.z;

        // Estimate the average number of particles per cell.
        float avgParticlesPerCell = particleCount / (float)gridCellCount;

        // Set maxParticlesPerCell to be a safety multiplier
        maxParticlesPerCell = Mathf.CeilToInt(avgParticlesPerCell * 25.0f);

        Debug.Log($"[SPH] Auto-adjusted spatial hash parameters:\n" +
                  $"  Cell Size: {cellSize}\n" +
                  $"  Grid Resolution: {gridResolution} (Total cells: {gridCellCount})\n" +
                  $"  Avg. Particles/Cell: {avgParticlesPerCell:F2}\n" +
                  $"  Max Particles per Cell: {maxParticlesPerCell}");
    }

    // ------------------------------------------------------
    // Setup & Initialization Methods
    // ------------------------------------------------------

    private void CreateParticles()
    {
        // Create the GPU buffer for particles.
        int stride = sizeof(float) * (3 + 3 + 3 + 1 + 1); // position(3) + velocity(3) + acceleration(3) + density(1) + pressure(1)
        particleBuffer = new ComputeBuffer(particleCount, stride);

        // Initialize particles with random positions within the spawn box.
        Particle[] particlesInit = new Particle[particleCount];
        for (int i = 0; i < particleCount; i++)
        {
            Vector3 halfSpawn = spawnSize * 0.5f;
            Vector3 randPos = new Vector3(
                spawnCenter.x + Random.Range(-halfSpawn.x, halfSpawn.x),
                spawnCenter.y + Random.Range(-halfSpawn.y, halfSpawn.y),
                spawnCenter.z + Random.Range(-halfSpawn.z, halfSpawn.z)
            );

            Particle p;
            p.position = randPos;
            p.velocity = Vector3.zero;
            p.acceleration = Vector3.zero;
            p.density = restDensity;
            p.pressure = 0f;
            particlesInit[i] = p;
        }
        particleBuffer.SetData(particlesInit);
    }

    private void CreateGrid()
    {
        gridCellCount = gridResolution.x * gridResolution.y * gridResolution.z;

        // Create a grid buffer (each cell can hold up to maxParticlesPerCell particle indices).
        gridBuffer = new ComputeBuffer(gridCellCount * maxParticlesPerCell, sizeof(int));

        // Create a buffer for grid cell indices (counts for each cell).
        gridIndicesBuffer = new ComputeBuffer(gridCellCount, sizeof(int));

        // Initialize grid buffers on the CPU.
        int[] gridData = new int[gridCellCount * maxParticlesPerCell];
        for (int i = 0; i < gridData.Length; i++)
            gridData[i] = -1;
        gridBuffer.SetData(gridData);

        int[] gridIndices = new int[gridCellCount];
        for (int i = 0; i < gridIndices.Length; i++)
            gridIndices[i] = 0;
        gridIndicesBuffer.SetData(gridIndices);
    }

    private void GetKernelIDs()
    {
        kernel_Clear = sphCompute.FindKernel("CS_Clear");
        kernel_ClearGrid = sphCompute.FindKernel("CS_ClearGrid");
        kernel_SpatialHash = sphCompute.FindKernel("CS_SpatialHash");
        kernel_DensityPressure = sphCompute.FindKernel("CS_DensityPressure");
        kernel_ComputeForces = sphCompute.FindKernel("CS_ComputeForces");
        kernel_XSPH = sphCompute.FindKernel("CS_XSPH");
        kernel_VvHalfStep = sphCompute.FindKernel("CS_VV_HalfStep");
        kernel_VvFullStep = sphCompute.FindKernel("CS_VV_FullStep");
        kernel_Boundaries = sphCompute.FindKernel("CS_Boundaries");
        kernel_ObstacleCollision = sphCompute.FindKernel("CS_ObstacleCollision");
        kernel_VelocityDamping = sphCompute.FindKernel("CS_VelocityDamping"); // New kernel

        if (kernel_Clear == -1 || kernel_ClearGrid == -1 || kernel_SpatialHash == -1 ||
            kernel_DensityPressure == -1 || kernel_ComputeForces == -1 || kernel_XSPH == -1 ||
            kernel_VvHalfStep == -1 || kernel_VvFullStep == -1 || kernel_Boundaries == -1 ||
            kernel_ObstacleCollision == -1 || kernel_VelocityDamping == -1)
        {
            Debug.LogError("One or more compute shader kernels were not found. Please check your kernel names.");
        }
        else
        {
            Debug.Log("Compute shader kernels found successfully.");
        }
    }

    private void SetComputeParams()
    {
        sphCompute.SetInt("_ParticleCount", particleCount);
        sphCompute.SetFloat("_ParticleMass", particleMass);
        sphCompute.SetFloat("_RestDensity", restDensity);
        sphCompute.SetFloat("_Stiffness", stiffness);
        sphCompute.SetFloat("_Viscosity", viscosity);
        sphCompute.SetFloat("_SmoothingRadius", smoothingRadius);
        sphCompute.SetFloat("_Gravity", gravity);
        sphCompute.SetFloat("_SurfaceTensionCoefficient", surfaceTensionCoefficient);
        sphCompute.SetFloat("_XSPHEpsilon", xsphEpsilon);
        sphCompute.SetFloat("_ParticleCollisionDamping", particleCollisionDamping);
        sphCompute.SetFloat("_ObstacleRepulsionStiffness", obstacleRepulsionStiffness);

        // Boundaries
        sphCompute.SetVector("_BoundsCenter", boundsCenter);
        sphCompute.SetVector("_BoundsSize", boundsSize);

        // Spawn box
        sphCompute.SetVector("_SpawnCenter", spawnCenter);
        sphCompute.SetVector("_SpawnSize", spawnSize);

        // Spatial hashing parameters
        sphCompute.SetInt("_MaxParticlesPerCell", maxParticlesPerCell);
        sphCompute.SetVector("_GridResolution", new Vector4(gridResolution.x, gridResolution.y, gridResolution.z, 0f));
        sphCompute.SetFloat("_NeighborSearchRadius", neighborSearchRadius);
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
        sphCompute.SetBuffer(kernel_ClearGrid, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_SpatialHash, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_ComputeForces, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_XSPH, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VvHalfStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VvFullStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_Boundaries, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_ObstacleCollision, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VelocityDamping, "Particles", particleBuffer); // New kernel

        sphCompute.SetBuffer(kernel_ClearGrid, "Grid", gridBuffer);
        sphCompute.SetBuffer(kernel_ClearGrid, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_SpatialHash, "Grid", gridBuffer);
        sphCompute.SetBuffer(kernel_SpatialHash, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "Grid", gridBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_ComputeForces, "Grid", gridBuffer);
        sphCompute.SetBuffer(kernel_ComputeForces, "GridIndices", gridIndicesBuffer);
        sphCompute.SetBuffer(kernel_XSPH, "Grid", gridBuffer);
        sphCompute.SetBuffer(kernel_XSPH, "GridIndices", gridIndicesBuffer);
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

    // ------------------------------------------------------------------
    // Public Accessors for External Scripts (e.g., FluidRayMarching.cs)
    // ------------------------------------------------------------------
    public ComputeBuffer GetParticleBuffer()
    {
        return particleBuffer;
    }

    public int ParticleCount
    {
        get { return particleCount; }
    }
}
