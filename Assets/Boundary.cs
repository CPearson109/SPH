using UnityEngine;
using System.Runtime.InteropServices;

[RequireComponent(typeof(LineRenderer))]
public class BoundaryRenderer : MonoBehaviour
{
    [Tooltip("Reference to the SPH simulation script.")]
    public SPH sph;

    private LineRenderer lineRenderer;
    // There are 12 edges (2 points per edge = 24 points)
    private Vector3[] edgePoints = new Vector3[24];

    // Define a struct matching the particle layout on the GPU.
    // It must match the layout in your compute shader:
    // Vector3 position, Vector3 velocity, Vector3 acceleration, float density, float pressure.
    [StructLayout(LayoutKind.Sequential)]
    public struct ParticleData
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 acceleration;
        public float density;
        public float pressure;
    }

    // Epsilon tolerance for collision detection in local space
    public float faceEpsilon = 0.05f;

    void Awake()
    {
        lineRenderer = GetComponent<LineRenderer>();
        lineRenderer.loop = false;
        // Use a basic material (you can change this as needed)
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.widthMultiplier = 0.05f;
        lineRenderer.positionCount = edgePoints.Length;
    }

    void Update()
    {
        if (sph == null || sph.boundaryCube == null)
            return;

        Transform boundaryTrans = sph.boundaryCube;

        // Define the 8 local corners of a unit cube (centered at origin).
        // In local space, a unit cube runs from -0.5 to +0.5 on each axis.
        Vector3[] localCorners = new Vector3[8]
        {
            new Vector3(-0.5f, -0.5f, -0.5f),
            new Vector3( 0.5f, -0.5f, -0.5f),
            new Vector3( 0.5f, -0.5f,  0.5f),
            new Vector3(-0.5f, -0.5f,  0.5f),
            new Vector3(-0.5f,  0.5f, -0.5f),
            new Vector3( 0.5f,  0.5f, -0.5f),
            new Vector3( 0.5f,  0.5f,  0.5f),
            new Vector3(-0.5f,  0.5f,  0.5f)
        };

        // Transform local corners into world space.
        Vector3[] worldCorners = new Vector3[8];
        for (int i = 0; i < 8; i++)
        {
            worldCorners[i] = boundaryTrans.TransformPoint(localCorners[i]);
        }

        // Print out the boundary corners (once per frame).
        string cornersMessage = "Boundary corners:\n";
        for (int i = 0; i < 8; i++)
        {
            cornersMessage += $"Corner {i}: {worldCorners[i]}\n";
        }
        Debug.Log(cornersMessage);

        // Build the edge points (for drawing with the LineRenderer).
        int p = 0;
        // Bottom face edges.
        edgePoints[p++] = worldCorners[0]; edgePoints[p++] = worldCorners[1];
        edgePoints[p++] = worldCorners[1]; edgePoints[p++] = worldCorners[2];
        edgePoints[p++] = worldCorners[2]; edgePoints[p++] = worldCorners[3];
        edgePoints[p++] = worldCorners[3]; edgePoints[p++] = worldCorners[0];
        // Top face edges.
        edgePoints[p++] = worldCorners[4]; edgePoints[p++] = worldCorners[5];
        edgePoints[p++] = worldCorners[5]; edgePoints[p++] = worldCorners[6];
        edgePoints[p++] = worldCorners[6]; edgePoints[p++] = worldCorners[7];
        edgePoints[p++] = worldCorners[7]; edgePoints[p++] = worldCorners[4];
        // Vertical edges.
        edgePoints[p++] = worldCorners[0]; edgePoints[p++] = worldCorners[4];
        edgePoints[p++] = worldCorners[1]; edgePoints[p++] = worldCorners[5];
        edgePoints[p++] = worldCorners[2]; edgePoints[p++] = worldCorners[6];
        edgePoints[p++] = worldCorners[3]; edgePoints[p++] = worldCorners[7];

        lineRenderer.positionCount = edgePoints.Length;
        lineRenderer.SetPositions(edgePoints);

        // Removed CPU-based per-particle collision checking for efficiency.
    }
}
