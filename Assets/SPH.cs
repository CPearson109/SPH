using UnityEngine;

public class SPH : MonoBehaviour
{
    [Header("Particle Settings")]
    // Removed global controls for particleMass, restDensity, and viscosity.
    // These are now controlled per-particle via SpawnBox.
    public float soundSpeed = 20f;
    public float gamma = 7f;
    public float stiffness = 800f; // (Not used in current computations)
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
    Optimized Fixed-Radius Nearest Neighbor Search (NNS) Integration:
    
    This simulation uses an advanced optimization technique for neighbor search in SPH:
    - The simulation space is partitioned into uniform grid cells.
    - Each particle is assigned to a grid cell based on its position.
    - Atomic operations are used in the compute shader (CS_BuildGrid) to insert particles into cells,
      achieving a counting sort–based approach that avoids the overhead of global sorting.
    - During neighbor search (density, pressure, and force calculations), only particles in adjacent cells
      are processed, reducing the computational complexity from O(n^2) to roughly O(n * k), where k is the average number of neighbors.
    
    This approach significantly enhances performance, enabling real-time simulations with millions of particles.
    */

    // ---------------------------
    // Internal Data Structures
    // ---------------------------
    // The Particle structure now holds:
    // - position (3), velocity (3), acceleration (3),
    // - density (1), pressure (1),
    // - restDensity (1), viscosity (1), mass (1) ← (all controlled by SpawnBox),
    // - color (4)
    // Total: 18 floats.
    struct Particle
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 acceleration;
        public float density;
        public float pressure;
        public float restDensity; // controlled by SpawnBox
        public float viscosity;   // controlled by SpawnBox
        public float mass;        // controlled by SpawnBox
        public Color color;       // for multiphase visualization
    }

    private ComputeBuffer particleBuffer;
    private ComputeBuffer gridCountsBuffer;
    private ComputeBuffer gridIndicesBuffer;

    // CPU-side arrays to help update static per-particle properties at runtime.
    private Particle[] particleArray;
    // This array tracks which SpawnBox each particle came from.
    private int[] particleSpawnBoxIndices;

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

        // Initialize CPU-side arrays.
        particleArray = new Particle[particleCount];
        particleSpawnBoxIndices = new int[particleCount];

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

        // Update each particle’s static properties (restDensity, viscosity, mass, color)
        // based on its originating SpawnBox. This ensures that any runtime changes in a SpawnBox's settings
        // are applied in real time.
        UpdateParticleStaticProperties();

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
        // Each particle has 18 floats (see Particle struct).
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
                // Set the static properties from the SpawnBox.
                p.restDensity = sb.restDensity;
                p.viscosity = sb.viscosity;
                p.mass = sb.particleMass;
                p.color = sb.particleColor;
                // Initialize dynamic properties.
                p.density = p.restDensity;
                p.pressure = 0f;
                particlesInit[index] = p;
                // Record which SpawnBox this particle came from.
                particleSpawnBoxIndices[index] = sbIndex;
                index++;
            }
        }
        particleBuffer.SetData(particlesInit);
        // Store a copy on the CPU for runtime updates.
        System.Array.Copy(particlesInit, particleArray, particleCount);
    }

    // This method updates the static properties (restDensity, viscosity, mass, color)
    // of each particle based on the current SpawnBox settings.
    private void UpdateParticleStaticProperties()
    {
        // Retrieve the current particle data from the GPU.
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
        // Write the updated static values back to the GPU.
        particleBuffer.SetData(particleArray);
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
        // Removed global _ParticleMass, _RestDensity, and _Viscosity; these are now per-particle.
        sphCompute.SetFloat("_SoundSpeed", soundSpeed);
        sphCompute.SetFloat("_Gamma", gamma);
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

    public ComputeBuffer GetParticleBuffer() { return particleBuffer; }
    public int ParticleCount { get { return particleCount; } }
}
