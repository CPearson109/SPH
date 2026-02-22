Shader "Custom/ParticleWaterRealistic_PerParticleColor_Opacified"
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

        _VelocityColor("High Velocity Additive Tint", Color) = (1, 1, 1, 1)
        _VelocityScale("Velocity Sensitivity", Float) = 1.0
        _EmissionStrength("Glow Strength", Float) = 3.0

            // Widened thresholds to prevent surface flickering
            _DensityThresholdMin("Surface Density Min", Float) = 1.15
            _DensityThresholdMax("Surface Density Max", Float) = 1.45
    }
        SubShader
        {
            Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }
            GrabPass { "_WaterGrabTex" }

            Pass
            {
                Cull Off
                ZWrite On
                ZTest LEqual
                Blend SrcAlpha OneMinusSrcAlpha

                CGPROGRAM
                #pragma vertex vert
                #pragma geometry geom
                #pragma fragment frag
                #pragma target 4.0
                #pragma multi_compile_fwdbase

                #include "UnityCG.cginc"
                #include "Lighting.cginc"

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
                    float4 color;
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

                float4 _VelocityColor;
                float _VelocityScale;
                float _EmissionStrength;

                float _DensityThresholdMin;
                float _DensityThresholdMax;

                struct appdata
                {
                    uint vertexID : SV_VertexID;
                };

                struct v2g
                {
                    float3 worldPos : TEXCOORD0;
                    float4 color    : COLOR0;
                };

                struct g2f
                {
                    float4 pos          : SV_POSITION;
                    float2 uv           : TEXCOORD0;
                    float3 worldPos     : TEXCOORD1;
                    float3 sphereNormal : TEXCOORD2;
                    float4 grabUV       : TEXCOORD3;
                    float4 color        : COLOR0;
                };

                static const float2 quadOffsets[4] =
                {
                    float2(-1,  1),
                    float2(1,  1),
                    float2(-1, -1),
                    float2(1, -1)
                };

                v2g vert(appdata v)
                {
                    v2g o;
                    Particle p = _ParticleBuffer[v.vertexID];
                    o.worldPos = p.position;

                    float speed = length(p.velocity);
                    float speedFactor = smoothstep(0.01, 2.0, speed * _VelocityScale);

                    float densityRatio = p.density / p.restDensity;
                    float surfaceMask = 1.0 - smoothstep(_DensityThresholdMin, _DensityThresholdMax, densityRatio);

                    speedFactor *= surfaceMask;

                    float3 dullColor = p.color.rgb * 0.2;
                    float3 brightColor = p.color.rgb + (_VelocityColor.rgb * _EmissionStrength);

                    o.color.rgb = lerp(dullColor, brightColor, speedFactor);
                    o.color.a = p.color.a;

                    return o;
                }

                [maxvertexcount(4)]
                void geom(point v2g input[1], inout TriangleStream<g2f> triStream)
                {
                    float3 center = input[0].worldPos;
                    float3 viewDir = normalize(center - _WorldSpaceCameraPos);

                    float3 up = abs(dot(float3(0,1,0), viewDir)) > 0.99 ? float3(1,0,0) : float3(0,1,0);
                    float3 right = normalize(cross(up, viewDir));
                    up = cross(viewDir, right);

                    right *= _SphereRadius;
                    up *= _SphereRadius;

                    for (int i = 0; i < 4; i++)
                    {
                        g2f o;
                        float2 offset = quadOffsets[i];
                        float3 posWorld = center + right * offset.x + up * offset.y;
                        o.pos = UnityWorldToClipPos(float4(posWorld, 1));
                        o.uv = offset;
                        o.worldPos = posWorld;
                        o.grabUV = ComputeGrabScreenPos(o.pos);

                        float r2 = saturate(dot(offset, offset));
                        float z = sqrt(1.0 - r2);
                        float3 localNormal = float3(offset.x, offset.y, z);

                        float3 worldNormal = normalize(
                            (right / _SphereRadius) * localNormal.x +
                            (up / _SphereRadius) * localNormal.y +
                            viewDir * localNormal.z
                        );

                        float wave = sin(_Time.y * _WaveSpeed + posWorld.x * _WaveScale)
                                   * cos(_Time.y * _WaveSpeed + posWorld.z * _WaveScale) * 0.1;
                        worldNormal += float3(wave, 0, wave);
                        o.sphereNormal = normalize(worldNormal);

                        o.color = input[0].color;
                        triStream.Append(o);
                    }
                }

                half4 frag(g2f i) : SV_Target
                {
                    float r = length(i.uv);
                    if (r > 1.0) discard;

                    float edgeAlpha = 1.0 - smoothstep(0.9, 1.0, r);
                    float mask = 1.0 - smoothstep(1.0 - _EdgeSoftness, 1.0, r);
                    float3 normal = normalize(i.sphereNormal);
                    float3 viewDir = normalize(_WorldSpaceCameraPos - i.worldPos);

                    float fresnel = pow(1.0 - saturate(dot(viewDir, normal)), _FresnelPower);
                    float3 reflection = texCUBE(_Cube, reflect(-viewDir, normal)).rgb;
                    float2 refractOffset = normal.xy * _RefractionStrength * edgeAlpha;

                    float4 grabUV = i.grabUV;
                    grabUV.xy += refractOffset;
                    float3 refraction = tex2Dproj(_WaterGrabTex, UNITY_PROJ_COORD(grabUV)).rgb;

                    float3 rrCombined = lerp(refraction, reflection, fresnel * _Reflectivity);
                    float3 baseColor = i.color.rgb;

                    float3 finalColor = lerp(rrCombined, baseColor, 0.6);

                    float3 lightDir = normalize(_WorldSpaceLightPos0.xyz);
                    float3 halfDir = normalize(lightDir + viewDir);
                    float spec = pow(saturate(dot(normal, halfDir)), _Shininess);
                    finalColor += _SpecularColor.rgb * spec * _Reflectivity;

                    float alpha = i.color.a * edgeAlpha * mask;

                    return half4(finalColor, alpha);
                }
                ENDCG
            }
        }
            FallBack "Diffuse"
}