using UnityEngine;

public class SPH : MonoBehaviour
{
    [Header("Particle Settings")]
    public int particleCount = 5000;
    public float particleMass = 0.8f;
    public float restDensity = 1000f;
    public float stiffness = 5000f;
    public float viscosity = 0.1f;
    public float smoothingRadius = 0.3f;
    public float gravity = -9.81f;

    [Header("Time Settings")]
    public float timeStep = 0.003f;
    public int subSteps = 4;

    [Header("Surface Tension")]
    public float surfaceTensionCoefficient = 0.1f;

    [Header("XSPH Settings")]
    public float xsphEpsilon = 0.5f;

    [Header("Boundary Settings")]
    public Vector3 boundsCenter = Vector3.zero;
    public Vector3 boundsSize = new Vector3(5f, 5f, 5f);

    [Header("Spawn Box Settings")]
    public Vector3 spawnCenter = new Vector3(0, 2, 0);
    public Vector3 spawnSize = new Vector3(2f, 2f, 2f);

    [Header("Obstacle Settings")]
    public Collider obstacleCollider;
    public float obstacleRepulsionStiffness = 5000f;
    public float particleCollisionDamping = 0.9f;

    [Header("Rendering & Debug Settings")]
    public Material fluidMaterial;
    public float gizmoParticleRadius = 0.1f;

    [Header("Compute Shader")]
    public ComputeShader sphCompute;

    // Public readonly property that other scripts (e.g. FluidRayMarching) can use
    public int ParticleCount => particleCount;

    // Expose the internal ComputeBuffer via a getter if needed in another script
    public ComputeBuffer GetParticleBuffer() => particleBuffer;

    struct Particle
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 acceleration;
        public float density;
        public float pressure;
    }

    private ComputeBuffer particleBuffer;
    private Particle[] particles;
    private const int THREAD_GROUP_SIZE = 256;

    // Kernels
    private int kernel_Clear;
    private int kernel_DensityPressure;
    private int kernel_ComputeForces;
    private int kernel_XSPH;
    private int kernel_VV_HalfStep;   // <--- Velocity Verlet half-step
    private int kernel_VV_FullStep;   // <--- Velocity Verlet final-step
    private int kernel_Boundaries;
    private int kernel_ObstacleCollision;

    void Start()
    {
        Application.targetFrameRate = 30;

        // Initialize particles
        particles = new Particle[particleCount];
        for (int i = 0; i < particleCount; i++)
        {
            particles[i].position = new Vector3(
                spawnCenter.x + Random.Range(-spawnSize.x * 0.5f, spawnSize.x * 0.5f),
                spawnCenter.y + Random.Range(-spawnSize.y * 0.5f, spawnSize.y * 0.5f),
                spawnCenter.z + Random.Range(-spawnSize.z * 0.5f, spawnSize.z * 0.5f)
            );
            particles[i].velocity = Vector3.zero;
            particles[i].acceleration = Vector3.zero;
            particles[i].density = restDensity;
            particles[i].pressure = 0f;
        }

        // Create and populate the particle buffer
        int stride = sizeof(float) * (3 + 3 + 3 + 1 + 1); // position, velocity, acceleration, density, pressure
        particleBuffer = new ComputeBuffer(particleCount, stride);
        particleBuffer.SetData(particles);

        // Find kernels in the compute shader
        kernel_Clear = sphCompute.FindKernel("CS_Clear");
        kernel_DensityPressure = sphCompute.FindKernel("CS_DensityPressure");
        kernel_ComputeForces = sphCompute.FindKernel("CS_ComputeForces");
        kernel_XSPH = sphCompute.FindKernel("CS_XSPH");
        kernel_VV_HalfStep = sphCompute.FindKernel("CS_VV_HalfStep");  // <---
        kernel_VV_FullStep = sphCompute.FindKernel("CS_VV_FullStep");  // <---
        kernel_Boundaries = sphCompute.FindKernel("CS_Boundaries");
        kernel_ObstacleCollision = sphCompute.FindKernel("CS_ObstacleCollision");

        // Set initial parameters on the compute shader
        UpdateComputeShaderParameters();

        // Assign the particle buffer to all kernels
        sphCompute.SetBuffer(kernel_Clear, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_DensityPressure, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_ComputeForces, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_XSPH, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VV_HalfStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_VV_FullStep, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_Boundaries, "Particles", particleBuffer);
        sphCompute.SetBuffer(kernel_ObstacleCollision, "Particles", particleBuffer);

        // If you have a material for rendering, let it access the particle buffer
        if (fluidMaterial != null)
        {
            fluidMaterial.SetBuffer("_ParticleBuffer", particleBuffer);
            fluidMaterial.SetInt("_NumParticles", particleCount);
            fluidMaterial.SetFloat("_ParticleRadius", smoothingRadius);
        }
    }

    void Update()
    {
        UpdateObstacleParameters();
        UpdateComputeShaderParameters(); // Re-apply dynamic parameters (in case changed in Inspector)

        float dtSub = timeStep / subSteps;
        for (int i = 0; i < subSteps; i++)
        {
            sphCompute.SetFloat("_DeltaTime", dtSub);

            // --- Velocity Verlet Half-step ---
            DispatchKernel(kernel_VV_HalfStep);

            // 1) Clear old density/pressure/acceleration
            DispatchKernel(kernel_Clear);

            // 2) Compute new density & pressure
            DispatchKernel(kernel_DensityPressure);

            // 3) Compute new forces => sets new p.acceleration
            DispatchKernel(kernel_ComputeForces);

            // 4) Optionally do XSPH velocity smoothing
            DispatchKernel(kernel_XSPH);

            // --- Velocity Verlet Final-step ---
            DispatchKernel(kernel_VV_FullStep);

            // 5) Apply boundaries and obstacle collisions
            DispatchKernel(kernel_Boundaries);
            DispatchKernel(kernel_ObstacleCollision);
        }

        // Read back data if needed for debugging or CPU side logic
        particleBuffer.GetData(particles);
    }

    /// <summary>
    /// Re-applies any parameters to the ComputeShader that might have changed in the Inspector.
    /// </summary>
    void UpdateComputeShaderParameters()
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

    /// <summary>
    /// Updates obstacle parameters if there is a SphereCollider used as an obstacle.
    /// </summary>
    void UpdateObstacleParameters()
    {
        if (obstacleCollider != null)
        {
            SphereCollider sphere = obstacleCollider as SphereCollider;
            if (sphere != null)
            {
                Vector3 obstaclePos = obstacleCollider.transform.position;
                sphCompute.SetVector("_ObstaclePos", obstaclePos);
                float scaledRadius = sphere.radius * obstacleCollider.transform.lossyScale.x;
                sphCompute.SetFloat("_ObstacleRadius", scaledRadius);
                return;
            }
        }

        // Default if no valid obstacle
        sphCompute.SetVector("_ObstaclePos", Vector3.zero);
        sphCompute.SetFloat("_ObstacleRadius", 0f);
    }

    void DispatchKernel(int kernelIndex)
    {
        int threadGroups = Mathf.CeilToInt((float)particleCount / THREAD_GROUP_SIZE);
        sphCompute.Dispatch(kernelIndex, threadGroups, 1, 1);
    }

    void OnDestroy()
    {
        if (particleBuffer != null)
            particleBuffer.Release();
    }

    /// <summary>
    /// Optional: Visualize bounding box and particles as Gizmos in the SceneView.
    /// </summary>
    void OnDrawGizmos()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(boundsCenter, boundsSize);

        Gizmos.color = Color.blue;
        if (particles != null)
        {
            foreach (var p in particles)
            {
                Gizmos.DrawSphere(p.position, gizmoParticleRadius);
            }
        }
    }
}
