Shader "Custom/ParticleFluid"
{
    Properties
    {
        _ParticleColor("Particle Color", Color) = (1, 1, 1, 1)
        _SphereRadius("Sphere Radius (world units)", Float) = 0.2
        _LightDir("Light Direction", Vector) = (0, 0, -1, 0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Transparent" }
        Pass
        {
            Cull Off
            ZWrite Off
            ZTest LEqual
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma geometry geom
            #pragma fragment frag
            #pragma target 4.0

            #include "UnityCG.cginc"

            // Updated Particle structure to match the compute shader.
            struct Particle
            {
                float3 position;
                float3 velocity;
                float3 acceleration;
                float density;
                float pressure;
                float restDensity; // per-particle rest density
                float viscosity;   // per-particle viscosity (dummy field for rendering)
            };

            StructuredBuffer<Particle> _ParticleBuffer;
            int _NumParticles;

            fixed4 _ParticleColor;
            float _SphereRadius;
            float4 _LightDir;

            struct appdata
            {
                uint vertexID : SV_VertexID;
            };

            struct v2g
            {
                float3 worldPos : TEXCOORD0;
                fixed4 color : COLOR0;
            };

            v2g vert(appdata v)
            {
                v2g o;
                Particle p = _ParticleBuffer[v.vertexID];
                o.worldPos = p.position;
                o.color = _ParticleColor;
                return o;
            }

            struct g2f
            {
                float4 pos : SV_POSITION;
                float2 uv  : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
                fixed4 color : COLOR0;
            };

            [maxvertexcount(6)]
            void geom(point v2g input[1], inout TriangleStream<g2f> triStream)
            {
                float3 center = input[0].worldPos;
                float3 viewDir = normalize(center - _WorldSpaceCameraPos);
                float3 up = float3(0, 1, 0);
                if (abs(dot(up, viewDir)) > 0.99)
                    up = float3(1, 0, 0);
                float3 right = normalize(cross(up, viewDir));
                up = cross(viewDir, right);

                right *= _SphereRadius;
                up *= _SphereRadius;

                float3 pos0 = center - right - up;
                float3 pos1 = center - right + up;
                float3 pos2 = center + right + up;
                float3 pos3 = center + right - up;

                g2f o;
                o.color = input[0].color;
                o.worldPos = center;

                o.pos = UnityWorldToClipPos(float4(pos0, 1.0));
                o.uv = float2(-1, -1);
                triStream.Append(o);

                o.pos = UnityWorldToClipPos(float4(pos1, 1.0));
                o.uv = float2(-1, 1);
                triStream.Append(o);

                o.pos = UnityWorldToClipPos(float4(pos2, 1.0));
                o.uv = float2(1, 1);
                triStream.Append(o);

                triStream.RestartStrip();

                o.pos = UnityWorldToClipPos(float4(pos0, 1.0));
                o.uv = float2(-1, -1);
                triStream.Append(o);

                o.pos = UnityWorldToClipPos(float4(pos2, 1.0));
                o.uv = float2(1, 1);
                triStream.Append(o);

                o.pos = UnityWorldToClipPos(float4(pos3, 1.0));
                o.uv = float2(1, -1);
                triStream.Append(o);
            }

            fixed4 frag(g2f i) : SV_Target
            {
                float2 uv = i.uv;
                if (dot(uv, uv) > 1.0)
                    discard;

                float z = sqrt(1.0 - dot(uv, uv));
                float3 normal = normalize(float3(uv.x, uv.y, z));
                float3 viewDir = normalize(_WorldSpaceCameraPos - i.worldPos);
                float fresnel = pow(1.0 - saturate(dot(normal, viewDir)), 3.0);
                float NdotL = saturate(dot(normal, normalize(_LightDir.xyz)));
                fixed3 diffuse = i.color.rgb * (0.2 + 0.8 * NdotL);
                float3 halfDir = normalize(viewDir + normalize(_LightDir.xyz));
                float spec = pow(saturate(dot(normal, halfDir)), 32.0);
                fixed3 fluidColor = lerp(diffuse, float3(1.0, 1.0, 1.0) * spec, fresnel);
                return fixed4(fluidColor, i.color.a);
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
}
