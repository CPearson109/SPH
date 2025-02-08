// Upgrade NOTE: commented out 'float3 _WorldSpaceCameraPos', a built-in variable

Shader "Custom/ParticleSphere"
{
    Properties
    {
        _ParticleColor("Particle Color", Color) = (1, 1, 1, 1)
        _SphereRadius("Sphere Radius (world units)", Float) = 0.2
        _LightDir("Light Direction", Vector) = (0, 0, -1, 0)
    }
        SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Transparent" }
        Pass
        {
            Cull Off
            ZWrite On
            ZTest LEqual
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
        // We need vertex, geometry, and fragment shaders.
        #pragma vertex vert
        #pragma geometry geom
        #pragma fragment frag
        #pragma target 4.0

        #include "UnityCG.cginc"

        //---------------------------------------------------------------------------
        // Data Structures & Uniforms
        //---------------------------------------------------------------------------
        // This struct must match the one in your compute shader.
        struct Particle
        {
            float3 position;
            float3 velocity;
            float3 acceleration;
            float density;
            float pressure;
        };

    // The structured buffer that holds particle data.
    StructuredBuffer<Particle> _ParticleBuffer;
    int _NumParticles;

    fixed4 _ParticleColor;
    float _SphereRadius;
    float4 _LightDir; // (x, y, z, unused)

    // Built-in camera position (set automatically by Unity)
    // float3 _WorldSpaceCameraPos;

    //---------------------------------------------------------------------------
    // Vertex Shader
    //---------------------------------------------------------------------------
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

    //---------------------------------------------------------------------------
    // Geometry Shader
    //---------------------------------------------------------------------------
    struct g2f
    {
        float4 pos : SV_POSITION;
        float2 uv  : TEXCOORD0; // Ranges from (-1,-1) to (1,1)
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

        o.pos = UnityObjectToClipPos(float4(pos0, 1.0));
        o.uv = float2(-1, -1);
        triStream.Append(o);

        o.pos = UnityObjectToClipPos(float4(pos1, 1.0));
        o.uv = float2(-1, 1);
        triStream.Append(o);

        o.pos = UnityObjectToClipPos(float4(pos2, 1.0));
        o.uv = float2(1, 1);
        triStream.Append(o);

        triStream.RestartStrip();

        o.pos = UnityObjectToClipPos(float4(pos0, 1.0));
        o.uv = float2(-1, -1);
        triStream.Append(o);

        o.pos = UnityObjectToClipPos(float4(pos2, 1.0));
        o.uv = float2(1, 1);
        triStream.Append(o);

        o.pos = UnityObjectToClipPos(float4(pos3, 1.0));
        o.uv = float2(1, -1);
        triStream.Append(o);
    }

    //---------------------------------------------------------------------------
    // Fragment Shader
    //---------------------------------------------------------------------------
    fixed4 frag(g2f i) : SV_Target
    {
        float2 uv = i.uv;
        if (dot(uv, uv) > 1.0)
            discard;

        float z = sqrt(1.0 - dot(uv, uv));
        float3 normal = normalize(float3(uv.x, uv.y, z));
        float NdotL = saturate(dot(normal, normalize(_LightDir.xyz)));
        fixed4 col = i.color * (0.2 + 0.8 * NdotL);
        return col;
    }
    ENDCG
}
}
FallBack "Diffuse"
}
