using UnityEngine;

public class CircularMotion : MonoBehaviour
{
    // The center of the circle
    public Vector3 center = Vector3.zero;
    // Radius of the circle
    public float radius = 5.0f;
    // Speed of the movement (radians per second)
    public float speed = 1.0f;

    void Update()
    {
        // Calculate the current angle based on time
        float angle = Time.time * speed;
        // Calculate x and z positions for a circle in the xz-plane
        float x = center.x + radius * Mathf.Cos(angle);
        float z = center.z + radius * Mathf.Sin(angle);
        // Update the position; keeping y constant (center.y)
        transform.position = new Vector3(x, center.y, z);
    }
}
