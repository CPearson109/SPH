using UnityEngine;

/// <summary>
/// Unified SPH with Spatial Hashing, Velocity Verlet, Surface Tension, and XSPH.
/// Optimized to handle up to 100k+ particles on modern GPUs.
/// </summary>
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
    [Tooltip("Resolution of the 3D grid used for spatial hashing.")]
    public Vector3Int gridResolution = new Vector3Int(64, 64, 64);

    [Tooltip("Maximum particles per grid cell.")]
    public int maxParticlesPerCell = 100;

    [Tooltip("Radius used to determine grid cell index (typically same as smoothingRadius).")]
    public float neighborSearchRadius = 0.3f;

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
    public float particleCollisionDamping = 0.9f;

    [Header("Rendering & Debug Settings")]
    [Tooltip("Material for rendering the fluid (draws particles as point sprites).")]
    public Material fluidMaterial;

    [Tooltip("Radius for drawing particles in editor Gizmos.")]
    public float gizmoParticleRadius = 0.1f;

    [Header("Compute Shaders")]
    public ComputeShader sphCompute; // Main SPH compute (all kernels)

    // ----- Internal Buffers & Data Structures -----
    struct Particle
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 acceleration;
        public float density;
        public float pressure;
    }

    private ComputeBuffer particleBuffer;      // All particles
    private ComputeBuffer gridBuffer;         // Stores up to maxParticlesPerCell indices per cell
    private ComputeBuffer gridIndicesBuffer;  // Stores how many particles are in each cell

    private Particle[] particlesCPU;          // CPU-side array (for Gizmos or debugging)
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

    // For dispatch
    private const int THREAD_GROUP_SIZE = 256;

    // --------------------------------------------------------------------
    void Start()
    {
        // Optionally limit frame rate for consistency
        Application.targetFrameRate = 60;

        // 1) Initialize particle array & GPU buffer
        CreateParticles();

        // 2) Initialize the grid buffers
        CreateGrid();

        // 3) Find compute kernels, set parameters
        GetKernelIDs();
        SetComputeParams();

        // 4) Bind the buffers to each kernel
        BindBuffers();

        // 5) (Optional) attach buffer to fluid rendering Material
        if (fluidMaterial != null)
        {
            fluidMaterial.SetBuffer("_ParticleBuffer", particleBuffer);
            fluidMaterial.SetInt("_NumParticles", particleCount);
            fluidMaterial.SetFloat("_ParticleRadius", smoothingRadius);
        }
    }

    private void Update()
    {
        // Update dynamic parameters (obstacle position, etc.)
        UpdateObstacle();

        // Update all parameters in the shader
        SetComputeParams();

        // Perform sub-stepping
        float dtSub = timeStep / subSteps;
        sphCompute.SetFloat("_DeltaTime", dtSub);

        // 1) Clear the grid
        DispatchCompute(kernel_ClearGrid, gridCellCount);

        // 2) Spatial hash to place particles in cells
        DispatchCompute(kernel_SpatialHash, particleCount);

        // 3) Multi-step solver
        for (int i = 0; i < subSteps; i++)
        {
            // 3a) Velocity Verlet half-step
            DispatchCompute(kernel_VvHalfStep, particleCount);

            // 3b) Clear per-particle accumulators
            DispatchCompute(kernel_Clear, particleCount);

            // 3c) Density & Pressure
            DispatchCompute(kernel_DensityPressure, particleCount);

            // 3d) Forces (pressure, viscosity, surface tension)
            DispatchCompute(kernel_ComputeForces, particleCount);

            // 3e) XSPH velocity correction
            DispatchCompute(kernel_XSPH, particleCount);

            // 3f) Velocity Verlet full-step
            DispatchCompute(kernel_VvFullStep, particleCount);

            // 3g) Boundary & obstacle collisions
            DispatchCompute(kernel_Boundaries, particleCount);
            DispatchCompute(kernel_ObstacleCollision, particleCount);
        }

        // Optional readback for debugging (expensive!)
        particleBuffer.GetData(particlesCPU);
    }

    private void OnDestroy()
    {
        // Release all GPU buffers
        particleBuffer?.Release();
        gridBuffer?.Release();
        gridIndicesBuffer?.Release();
    }

    private void OnDrawGizmos()
    {
        // Draw the bounding box
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(boundsCenter, boundsSize);

        // Draw the obstacle (if it’s a sphere)
        if (obstacleCollider is SphereCollider sphere)
        {
            Gizmos.color = Color.yellow;
            float scaledRadius = sphere.radius * obstacleCollider.transform.lossyScale.x;
            Gizmos.DrawWireSphere(obstacleCollider.transform.position, scaledRadius);
        }

        // Draw the particles (debug) – can be costly with large particleCount
        if (particlesCPU != null)
        {
            Gizmos.color = Color.blue;
            for (int i = 0; i < particlesCPU.Length; i++)
            {
                Gizmos.DrawSphere(particlesCPU[i].position, gizmoParticleRadius);
            }
        }
    }

    // ------------------------------------------------------
    // Setup & Initialization
    // ------------------------------------------------------

    private void CreateParticles()
    {
        particlesCPU = new Particle[particleCount];

        // Fill with random positions within the spawn box
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
            particlesCPU[i] = p;
        }

        // Create GPU buffer
        int stride = sizeof(float) * (3 + 3 + 3 + 1 + 1); // position(3) + velocity(3) + accel(3) + density(1) + pressure(1)
        particleBuffer = new ComputeBuffer(particleCount, stride);
        particleBuffer.SetData(particlesCPU);
    }

    private void CreateGrid()
    {
        // Total cells
        gridCellCount = gridResolution.x * gridResolution.y * gridResolution.z;

        // gridBuffer: each cell has 'maxParticlesPerCell' indices
        gridBuffer = new ComputeBuffer(gridCellCount * maxParticlesPerCell, sizeof(int));

        // gridIndicesBuffer: each cell has a single integer to count how many are used
        gridIndicesBuffer = new ComputeBuffer(gridCellCount, sizeof(int));

        // Initialize them on CPU
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
        // Grab kernel indices from the compute shader
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

        // Validation
        if (kernel_Clear == -1 || kernel_ClearGrid == -1 || kernel_SpatialHash == -1 ||
            kernel_DensityPressure == -1 || kernel_ComputeForces == -1 || kernel_XSPH == -1 ||
            kernel_VvHalfStep == -1 || kernel_VvFullStep == -1 || kernel_Boundaries == -1 ||
            kernel_ObstacleCollision == -1)
        {
            Debug.LogError("One or more kernels were not found in the Compute Shader. Check kernel names.");
        }
        else
        {
            Debug.Log($"Kernel Indices: Clear={kernel_Clear}, ClearGrid={kernel_ClearGrid}, SpatialHash={kernel_SpatialHash}, DensityPressure={kernel_DensityPressure}, ComputeForces={kernel_ComputeForces}, XSPH={kernel_XSPH}, VvHalfStep={kernel_VvHalfStep}, VvFullStep={kernel_VvFullStep}, Boundaries={kernel_Boundaries}, ObstacleCollision={kernel_ObstacleCollision}");
        }
    }

    private void SetComputeParams()
    {
        // Core parameters
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

        // Spatial hashing
        sphCompute.SetInt("_MaxParticlesPerCell", maxParticlesPerCell);
        sphCompute.SetVector("_GridResolution", new Vector4(gridResolution.x, gridResolution.y, gridResolution.z, 0f));
        sphCompute.SetFloat("_NeighborSearchRadius", neighborSearchRadius);
    }

    private void UpdateObstacle()
    {
        // If we have a sphere collider, update its position/radius
        if (obstacleCollider is SphereCollider sphere)
        {
            Vector3 pos = obstacleCollider.transform.position;
            float radius = sphere.radius * obstacleCollider.transform.lossyScale.x;
            sphCompute.SetVector("_ObstaclePos", pos);
            sphCompute.SetFloat("_ObstacleRadius", radius);
        }
        else
        {
            // No valid sphere
            sphCompute.SetVector("_ObstaclePos", Vector3.zero);
            sphCompute.SetFloat("_ObstacleRadius", 0f);
        }
    }

    private void BindBuffers()
    {
        // Particle buffer
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

        // Grid buffers
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

    // ------------------------------------------------------
    // Compute Dispatch Helper
    // ------------------------------------------------------

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

    // ------------------------------------------------------
    // Public Accessors
    // ------------------------------------------------------

    /// <summary>
    /// Returns the ComputeBuffer containing particle data.
    /// </summary>
    public ComputeBuffer GetParticleBuffer()
    {
        return particleBuffer;
    }

    /// <summary>
    /// Returns the total number of particles in the simulation.
    /// </summary>
    public int ParticleCount => particleCount;
}