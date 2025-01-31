using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FluidRayMarching : MonoBehaviour
{
    public ComputeShader raymarching;
    public Camera cam;

    List<ComputeBuffer> buffersToDispose = new List<ComputeBuffer>();

    public SPH sph; // Reference to the SPH script

    RenderTexture target;

    [Header("Params")]
    public float viewRadius = 0.5f;
    public float blendStrength = 1.0f;
    public Color waterColor = Color.blue;

    public Color ambientLight = Color.white;

    public Light lightSource;

    private bool render = false;

    void InitRenderTexture()
    {
        if (target == null || target.width != cam.pixelWidth || target.height != cam.pixelHeight)
        {
            if (target != null)
            {
                target.Release();
            }

            cam.depthTextureMode = DepthTextureMode.Depth;

            target = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            target.enableRandomWrite = true;
            target.Create();
        }
    }

    public void Begin()
    {
        // Ensure the render texture is initialized
        InitRenderTexture();

        // Set up the raymarching compute shader
        raymarching.SetBuffer(0, "particles", sph.GetParticleBuffer()); // Use particle buffer from SPH
        raymarching.SetInt("numParticles", sph.ParticleCount);          // Number of particles
        raymarching.SetFloat("particleRadius", viewRadius);             // Particle radius
        raymarching.SetFloat("blendStrength", blendStrength);           // Blend strength for rendering
        raymarching.SetVector("waterColor", waterColor);                // Fluid color
        raymarching.SetVector("_AmbientLight", ambientLight);           // Ambient light color
        raymarching.SetTextureFromGlobal(0, "_DepthTexture", "_CameraDepthTexture");

        render = true;
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (!render)
        {
            Begin();
        }

        if (render)
        {
            Debug.Assert(target != null, "RenderTexture is not created.");
            Debug.Log($"RenderTexture dimensions: {target.width} x {target.height}");


            raymarching.SetVector("_Light", lightSource.transform.forward); // Set light direction
            raymarching.SetTexture(0, "Source", source);                    // Input texture
            raymarching.SetTexture(0, "Destination", target);               // Render target
            raymarching.SetVector("_CameraPos", cam.transform.position);    // Camera position
            raymarching.SetMatrix("_CameraToWorld", cam.cameraToWorldMatrix); // Camera-to-world matrix
            raymarching.SetMatrix("_CameraInverseProjection", cam.projectionMatrix.inverse); // Inverse projection matrix

            int threadGroupsX = Mathf.CeilToInt(cam.pixelWidth / 8.0f);
            int threadGroupsY = Mathf.CeilToInt(cam.pixelHeight / 8.0f);
            raymarching.Dispatch(0, threadGroupsX, threadGroupsY, 1);       // Dispatch the compute shader

            Graphics.Blit(target, destination); // Copy the output to the screen
        }
    }

    void OnDestroy()
    {
        // Release the render texture
        if (target != null)
        {
            target.Release();
        }

        // Dispose of any buffers
        foreach (var buffer in buffersToDispose)
        {
            buffer.Release();
        }
    }
}
