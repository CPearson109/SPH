using UnityEngine;

[ExecuteAlways] // so that it works in edit mode too
public class SpawnBox : MonoBehaviour
{
    [Header("Spawn Box Settings")]
    [Tooltip("Number of particles to spawn from this box.")]
    public int particleCount = 1000;

    [Tooltip("Size of the spawn box.")]
    public Vector3 spawnSize = new Vector3(2f, 2f, 2f);

    [Tooltip("Rest density for particles spawned from this box.")]
    public float restDensity = 1000f;

    [Tooltip("Viscosity for particles spawned from this box.")]
    public float viscosity = 0.1f;

    // A convenient property: the center is just the GameObject's position.
    public Vector3 SpawnCenter => transform.position;

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(transform.position, spawnSize);
    }
}
