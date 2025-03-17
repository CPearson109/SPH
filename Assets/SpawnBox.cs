using UnityEngine;

[ExecuteAlways]
public class SpawnBox : MonoBehaviour
{
    [Header("Spawn Box Settings")]
    public int particleCount = 1000;
    public Vector3 spawnSize = new Vector3(2f, 2f, 2f);

    [Tooltip("Per-particle rest density.")]
    public float restDensity = 1000f;

    [Tooltip("Per-particle viscosity.")]
    public float viscosity = 0.1f;

    [Tooltip("Per-particle mass.")]
    public float particleMass = 0.03f;

    [Tooltip("Initial velocity for spawned particles.")]
    public Vector3 initialVelocity = Vector3.zero;

    [Tooltip("Particle color (alpha < 1 for blending).")]
    public Color particleColor = Color.white;

    public Vector3 SpawnCenter => transform.position;

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(transform.position, spawnSize);
    }
}
