﻿Shader "Custom/ParticleWaterRealistic_PerParticleColor_Opacified"
{
    Properties
    {
        _SphereRadius("Sphere Radius (world units)", Float) = 0.2
        _EdgeSoftness("Edge Softness", Range(0,1)) = 0.2
        _Cube("Environment Cubemap", Cube) = "" {}
        _FresnelPower("Fresnel Power", Float) = 5.0
        _Reflectivity("Reflectivity", Range(0,1)) = 0.8
        _RefractionStrength("Refraction Strength", Range(0,1)) = 0.1
        _SpecularColor("Specular Color", Color) = (1,1,1,1)
        _Shininess("Shininess", Float) = 128.0
        _WaveSpeed("Wave Speed", Float) = 1.0
        _WaveScale("Wave Scale", Float) = 0.5
    }
        SubShader
        {
            // Use transparent rendering order.
            Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }

            // Grab the refraction texture.
            GrabPass { "_WaterGrabTex" }

            Pass
            {
                // Set up blending, culling and depth settings.
                Cull Off
                ZWrite On          // Enable depth writes for proper sorting.
                ZTest LEqual       // Use depth testing.
                Blend SrcAlpha OneMinusSrcAlpha

                CGPROGRAM
                #pragma vertex vert
                #pragma geometry geom
                #pragma fragment frag
                #pragma target 4.0
                #pragma multi_compile_fwdbase

                #include "UnityCG.cginc"
                #include "Lighting.cginc"

            //-------------------------------------------------------------------------
            // Overview:
            // This shader renders individual fluid particles as spheres using
            // billboard geometry. It approximates a sphere’s normal for lighting,
            // and applies reflection, refraction and Fresnel effects.
            //
            // References:
            //   - Unity Compute Shaders and Graphics: https://docs.unity3d.com/Manual/SL-ShaderPrograms.html
            //   - Fresnel and refraction techniques in realtime rendering.
            //-------------------------------------------------------------------------

            // Particle structure (must match compute buffer layout).
            struct Particle
            {
                float3 position;
                float3 velocity;
                float3 acceleration;
                float density;
                float pressure;
                float restDensity;
                float viscosity;
                float mass;
                float4 color; // RGBA per-particle color.
            };

            StructuredBuffer<Particle> _ParticleBuffer;
            int _NumParticles;

            float _SphereRadius;
            float _EdgeSoftness;
            samplerCUBE _Cube;
            float _FresnelPower;
            float _Reflectivity;
            float _RefractionStrength;
            fixed4 _SpecularColor;
            float _Shininess;
            float _WaveSpeed;
            float _WaveScale;
            sampler2D _WaterGrabTex;

            // Vertex input: using the vertexID to index into the particle buffer.
            struct appdata
            {
                uint vertexID : SV_VertexID;
            };

            // Vertex-to-geometry structure.
            struct v2g
            {
                float3 worldPos : TEXCOORD0;
                float4 color    : COLOR0;
            };

            // Geometry-to-fragment structure.
            struct g2f
            {
                float4 pos          : SV_POSITION;
                float2 uv           : TEXCOORD0;
                float3 worldPos     : TEXCOORD1;
                float3 sphereNormal : TEXCOORD2;
                float4 grabUV       : TEXCOORD3;
                float4 color        : COLOR0;
            };

            // Pre-defined quad offsets for a billboard.
            static const float2 quadOffsets[6] =
            {
                float2(-1, -1),
                float2(-1,  1),
                float2(1,  1),
                float2(-1, -1),
                float2(1,  1),
                float2(1, -1)
            };

            // Vertex shader: retrieves particle data.
            v2g vert(appdata v)
            {
                v2g o;
                Particle p = _ParticleBuffer[v.vertexID];
                o.worldPos = p.position;
                o.color = p.color;
                return o;
            }

            // Geometry shader: builds a billboard quad for each particle.
            [maxvertexcount(6)]
            void geom(point v2g input[1], inout TriangleStream<g2f> triStream)
            {
                float3 center = input[0].worldPos;
                float3 viewDir = normalize(center - _WorldSpaceCameraPos);

                // Determine billboard basis vectors.
                float3 up = abs(dot(float3(0,1,0), viewDir)) > 0.99 ? float3(1,0,0) : float3(0,1,0);
                float3 right = normalize(cross(up, viewDir));
                up = cross(viewDir, right);

                right *= _SphereRadius;
                up *= _SphereRadius;

                // Create six vertices forming two triangles.
                for (int i = 0; i < 6; i++)
                {
                    g2f o;
                    float2 offset = quadOffsets[i];
                    float3 posWorld = center + right * offset.x + up * offset.y;
                    o.pos = UnityWorldToClipPos(float4(posWorld, 1));
                    o.uv = offset;
                    o.worldPos = posWorld;
                    o.grabUV = ComputeGrabScreenPos(o.pos);

                    // Approximate sphere normal from the offset.
                    float r2 = saturate(dot(offset, offset));
                    float z = sqrt(1.0 - r2);
                    float3 localNormal = float3(offset.x, offset.y, z);

                    // Transform the local normal into world space.
                    float3 worldNormal = normalize(
                        (right / _SphereRadius) * localNormal.x +
                        (up / _SphereRadius) * localNormal.y +
                        viewDir * localNormal.z
                    );

                    // Optional wave effect for surface detail.
                    float wave = sin(_Time.y * _WaveSpeed + posWorld.x * _WaveScale)
                               * cos(_Time.y * _WaveSpeed + posWorld.z * _WaveScale) * 0.1;
                    worldNormal += float3(wave, 0, wave);
                    o.sphereNormal = normalize(worldNormal);

                    o.color = input[0].color;
                    triStream.Append(o);
                }
            }

            // Fragment shader: computes final color, lighting, and blending.
            fixed4 frag(g2f i) : SV_Target
            {
                // Discard fragments outside the circular particle boundary.
                float r = length(i.uv);
                if (r > 1.0) discard;

                // Edge fade and soft blending.
                float edgeAlpha = 1.0 - smoothstep(0.9, 1.0, r);
                float mask = 1.0 - smoothstep(1.0 - _EdgeSoftness, 1.0, r);

                float3 normal = normalize(i.sphereNormal);
                float3 viewDir = normalize(_WorldSpaceCameraPos - i.worldPos);

                // Fresnel effect calculation.
                float fresnel = pow(1.0 - saturate(dot(viewDir, normal)), _FresnelPower);

                // Reflection from the environment cubemap.
                float3 reflection = texCUBE(_Cube, reflect(-viewDir, normal)).rgb;
                float2 refractOffset = normal.xy * _RefractionStrength * edgeAlpha;

                // Refraction using the grab texture.
                float4 grabUV = i.grabUV;
                grabUV.xy += refractOffset;
                float3 refraction = tex2Dproj(_WaterGrabTex, UNITY_PROJ_COORD(grabUV)).rgb;

                // Combine refraction and reflection based on Fresnel factor.
                float3 rrCombined = lerp(refraction, reflection, fresnel * _Reflectivity);

                // Blend with the particle's base color.
                float3 baseColor = i.color.rgb;
                float3 finalColor = lerp(rrCombined, baseColor, 0.3);

                // Add specular highlights.
                float3 lightDir = normalize(_WorldSpaceLightPos0.xyz);
                float3 halfDir = normalize(lightDir + viewDir);
                float spec = pow(saturate(dot(normal, halfDir)), _Shininess);
                finalColor += _SpecularColor.rgb * spec * _Reflectivity;

                // Compute final alpha value.
                float alpha = i.color.a * edgeAlpha * mask;

                return fixed4(finalColor, alpha);
            }
            ENDCG
        }
        }
            FallBack "Diffuse"
}
