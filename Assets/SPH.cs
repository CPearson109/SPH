using UnityEngine;

public class SPH : MonoBehaviour
{
    [Header("Particle Settings")]
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

    [Header("Axis-Aligned Boundary (Fallback)")]
    [Tooltip("Center of the simulation bounding box (used if no boundary cube is assigned).")]
    public Vector3 boundsCenter = Vector3.zero;
    [Tooltip("Size of the simulation bounding box (used if no boundary cube is assigned).")]
    public Vector3 boundsSize = new Vector3(10f, 10f, 10f);

    [Header("Boundary Cube")]
    [Tooltip("Assign a 3D object (e.g., a cube) whose edges define the simulation boundary. " +
             "When this object is moved/rotated/scaled at runtime, the particles will follow.")]
    public Transform boundaryCube;

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

    private ComputeBuffer particleBuffer;    // Buffer containing all particles.
    private ComputeBuffer gridCountsBuffer;    // Buffer holding particle counts per grid cell.
    private ComputeBuffer gridIndicesBuffer;   // Buffer holding particle indices per grid cell.

    // Grid parameters computed from the fallback axis–aligned boundary.
    private int gridResolutionX, gridResolutionY, gridResolutionZ;
    private int totalCells;
    private Vector3 gridMin; // Lower corner of the fallback boundary.

    // Kernel indices (set by FindKernel; names must match exactly).
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

    // Thread group size: must match [numthreads(256,1,1)] in the compute shader.
    private const int THREAD_GROUP_SIZE = 256;

    // Total number of particles.
    private int particleCount;

    void Start()
    {
        Application.targetFrameRate = 60;

        // If no spawn boxes are assigned, try to find them automatically.
        if (spawnBoxes == null || spawnBoxes.Length == 0)
        {
            spawnBoxes = FindObjectsOfType<SpawnBox>();
        }

        // Sum particle counts from all spawn boxes.
        particleCount = 0;
        foreach (var sb in spawnBoxes)
        {
            particleCount += sb.particleCount;
        }

        // 1) Initialize particles and create the GPU buffer.
        CreateParticles();

        // 2) Compute grid resolution using the fallback bounds and smoothing radius.
        Vector3 halfBounds = boundsSize * 0.5f;
        gridMin = boundsCenter - halfBounds;
        gridResolutionX = Mathf.CeilToInt(boundsSize.x / smoothingRadius);
        gridResolutionY = Mathf.CeilToInt(boundsSize.y / smoothingRadius);
        gridResolutionZ = Mathf.CeilToInt(boundsSize.z / smoothingRadius);
        totalCells = gridResolutionX * gridResolutionY * gridResolutionZ;

        // Create grid buffers.
        gridCountsBuffer = new ComputeBuffer(totalCells, sizeof(int));
        gridIndicesBuffer = new ComputeBuffer(totalCells * maxParticlesPerCell, sizeof(int));

        // 3) Get the kernel IDs and set simulation parameters.
        GetKernelIDs();
        SetComputeParams();

        // 4) Bind the particle and grid buffers.
        BindBuffers();
        if (fluidMaterial != null)
        {
            fluidMaterial.SetBuffer("_ParticleBuffer", particleBuffer);
            fluidMaterial.SetInt("_NumParticles", particleCount);
        }
    }

    void Update()
    {
        // Update dynamic parameters (such as the obstacle and boundary transforms).
        UpdateObstacle();
        SetComputeParams();
        BindBuffers();

        // Rebuild the spatial grid.
        DispatchCompute(kernel_ClearGrid, totalCells);
        DispatchCompute(kernel_BuildGrid, particleCount);

        // Set sub-step time.
        float dtSub = timeStep / subSteps;
        sphCompute.SetFloat("_DeltaTime", dtSub);

        // Run the multi-step solver.
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

    // Render the particles using the GPU.
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

    // Use Gizmos to visualize the simulation boundary.
    void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        if (boundaryCube != null)
        {
            // Draw a unit cube (centered at the origin) transformed by the boundaryCube's matrix.
            Gizmos.matrix = boundaryCube.localToWorldMatrix;
            Gizmos.DrawWireCube(Vector3.zero, Vector3.one);
            Gizmos.matrix = Matrix4x4.identity;
        }
        else
        {
            // Fallback: Draw the axis–aligned boundary.
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
        // The kernel names here must match exactly those in the compute shader.
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

        // Log the kernel IDs for debugging.
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
        sphCompute.SetFloat("_Stiffness", stiffness);
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
            Matrix4x4 boundaryInvMatrix = boundaryCube.worldToLocalMatrix;  // Precomputed inverse
            sphCompute.SetMatrix("_BoundaryMatrix", boundaryMatrix);
            sphCompute.SetMatrix("_BoundaryInvMatrix", boundaryInvMatrix);
            // Use a fixed half-extent corresponding to a unit cube.
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
