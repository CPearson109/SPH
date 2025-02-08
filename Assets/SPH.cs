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

    private ComputeBuffer particleBuffer;  // All particles

    // Kernel indices (only those that do not rely on spatial hashing)
    private int kernel_Clear;
    private int kernel_DensityPressure;
    private int kernel_ComputeForces;
    private int kernel_XSPH;
    private int kernel_VvHalfStep;
    private int kernel_VvFullStep;
    private int kernel_Boundaries;
    private int kernel_ObstacleCollision;
    private int kernel_VelocityDamping;

    // For dispatching compute threads
    private const int THREAD_GROUP_SIZE = 256;

    void Start()
    {
        // Optionally limit frame rate for consistency
        Application.targetFrameRate = 60;

        // 1) Initialize particles and GPU buffer
        CreateParticles();

        // 2) Get compute shader kernels and set simulation parameters
        GetKernelIDs();
        SetComputeParams();

        // 3) Bind the particle buffer to the compute shader and to the rendering material
        BindBuffers();

        if (fluidMaterial != null)
        {
            fluidMaterial.SetBuffer("_ParticleBuffer", particleBuffer);
            fluidMaterial.SetInt("_NumParticles", particleCount);
        }
    }

    void Update()
    {
        // update dynamic parameters)
        UpdateObstacle();
        SetComputeParams();

        // Rebind the particle buffer for each kernel
        BindBuffers();

        // Set sub-step time
        float dtSub = timeStep / subSteps;
        sphCompute.SetFloat("_DeltaTime", dtSub);

        // Run the multi-step solver
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


    // Render the particles on the GPU
    void OnRenderObject()
    {
        if (fluidMaterial != null)
        {
            fluidMaterial.SetPass(0);
            Graphics.DrawProceduralNow(MeshTopology.Points, particleCount);
        }
    }

    private void OnDestroy()
    {
        // Release the GPU buffer
        particleBuffer?.Release();
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

    private void GetKernelIDs()
    {
        kernel_Clear = sphCompute.FindKernel("CS_Clear");
        kernel_DensityPressure = sphCompute.FindKernel("CS_DensityPressure");
        kernel_ComputeForces = sphCompute.FindKernel("CS_ComputeForces");
        kernel_XSPH = sphCompute.FindKernel("CS_XSPH");
        kernel_VvHalfStep = sphCompute.FindKernel("CS_VV_HalfStep");
        kernel_VvFullStep = sphCompute.FindKernel("CS_VV_FullStep");
        kernel_Boundaries = sphCompute.FindKernel("CS_Boundaries");
        kernel_ObstacleCollision = sphCompute.FindKernel("CS_ObstacleCollision");
        kernel_VelocityDamping = sphCompute.FindKernel("CS_VelocityDamping");

        // Also, if a separate clear kernel exists, it should be found here.
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

        sphCompute.SetVector("_BoundsCenter", boundsCenter);
        sphCompute.SetVector("_BoundsSize", boundsSize);

        sphCompute.SetVector("_SpawnCenter", spawnCenter);
        sphCompute.SetVector("_SpawnSize", spawnSize);
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
        sphCompute.SetBuffer(kernel_DensityPressure, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_ComputeForces, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_XSPH, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VvHalfStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VvFullStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_Boundaries, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_ObstacleCollision, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VelocityDamping, "Particles", particleBuffer);
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
    // Public Accessors for External Scripts
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
