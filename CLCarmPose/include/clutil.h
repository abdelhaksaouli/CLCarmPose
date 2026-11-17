
#define BLOCK_SIZE 64
#define N_ANGLE 45
#define D_ANGLE0 (2.0f * (float)M_PI_F)/ (float)N_ANGLE

/***************************************************** Mat 3 x 3  **************************************/
typedef struct matrix3{
    float matrix[3][3];
} mat3;



inline mat3 zeroes3x3()
{
  mat3 m;
  m.matrix[0][0] = 0.0f;
  m.matrix[0][1] = 0.0f;
  m.matrix[0][2] = 0.0f;

  m.matrix[1][0] = 0.0f;
  m.matrix[1][1] = 0.0f;
  m.matrix[1][2] = 0.0f;

  m.matrix[2][0] = 0.0f;
  m.matrix[2][1] = 0.0f;
  m.matrix[2][2] = 0.0f;
  return m;
}

inline float3 mat3_mul_v3(mat3 m, float3 v) 
{
    float3 r0 = (float3)(m.matrix[0][0], m.matrix[0][1], m.matrix[0][2]);
    float3 r1 = (float3)(m.matrix[1][0], m.matrix[1][1], m.matrix[1][2]);
    float3 r2 = (float3)(m.matrix[2][0], m.matrix[2][1], m.matrix[2][2]);
    return (float3)(dot(v, r0), dot(v, r1), dot(v, r2));
}

/***************************************************** Mat 4 x 4  **************************************/
typedef struct matrix4{
    float matrix[4][4];
} mat4;


/* matrix constructor */

inline mat4 zeroes4x4()
{
    mat4 m;
    m.matrix[0][0] = 0.0f;
    m.matrix[0][1] = 0.0f;
    m.matrix[0][2] = 0.0f;
    m.matrix[0][3] = 0.0f;

    m.matrix[1][0] = 0.0f;
    m.matrix[1][1] = 0.0f;
    m.matrix[1][2] = 0.0f;
    m.matrix[1][3] = 0.0f;

    m.matrix[2][0] = 0.0f;
    m.matrix[2][1] = 0.0f;
    m.matrix[2][2] = 0.0f;
    m.matrix[2][3] = 0.0f;

    m.matrix[3][0] = 0.0f;
    m.matrix[3][1] = 0.0f;
    m.matrix[3][2] = 0.0f;
    m.matrix[3][3] = 0.0f;

  return m;
}

inline mat4 identity()
{
    mat4 m;
    m.matrix[0][0] = 1.0f;
    m.matrix[0][1] = 0.0f;
    m.matrix[0][2] = 0.0f;
    m.matrix[0][3] = 0.0f;

    m.matrix[1][0] = 0.0f;
    m.matrix[1][1] = 1.0f;
    m.matrix[1][2] = 0.0f;
    m.matrix[1][3] = 0.0f;

    m.matrix[2][0] = 0.0f;
    m.matrix[2][1] = 0.0f;
    m.matrix[2][2] = 1.0f;
    m.matrix[2][3] = 0.0f;

    m.matrix[3][0] = 0.0f;
    m.matrix[3][1] = 0.0f;
    m.matrix[3][2] = 0.0f;
    m.matrix[3][3] = 1.0f;

    return m;
}


inline float4 mat4_mul_v4(mat4 m, float4 v) 
{
    float4 r0 = (float4)(m.matrix[0][0], m.matrix[0][1], m.matrix[0][2], m.matrix[0][3]);
    float4 r1 = (float4)(m.matrix[1][0], m.matrix[1][1], m.matrix[1][2], m.matrix[1][3]);
    float4 r2 = (float4)(m.matrix[2][0], m.matrix[2][1], m.matrix[2][2], m.matrix[2][3]);
    float4 r3 = (float4)(m.matrix[3][0], m.matrix[3][1], m.matrix[3][2], m.matrix[3][3]);

    return (float4)(dot(v, r0), dot(v, r1), dot(v, r2), dot(v, r3));
}

inline mat4 multiply4x4(mat4 A, mat4 B)
{
    mat4 C = zeroes4x4();

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                C.matrix[i][j] += A.matrix[i][k] * B.matrix[k][j];
            }
        }
    }

    return C;
}


inline mat4 invert4x4(mat4 a)
{
    // T = [R t;0 1]; inv(T) = [R' -R'*t;0 1]; R*R' = I
    
    mat4 invert;

    //R'
    invert.matrix[0][0] = a.matrix[0][0];
    invert.matrix[0][1] = a.matrix[1][0];
    invert.matrix[0][2] = a.matrix[2][0];

    invert.matrix[1][0] = a.matrix[0][1];
    invert.matrix[1][1] = a.matrix[1][1];
    invert.matrix[1][2] = a.matrix[2][1];

    invert.matrix[2][0] = a.matrix[0][2];
    invert.matrix[2][1] = a.matrix[1][2];
    invert.matrix[2][2] = a.matrix[2][2];


    //-R'
    mat3  neg_Rtranspose;

    neg_Rtranspose.matrix[0][0] = -invert.matrix[0][0];
    neg_Rtranspose.matrix[0][1] = -invert.matrix[0][1];
    neg_Rtranspose.matrix[0][2] = -invert.matrix[0][2];

    neg_Rtranspose.matrix[1][0] = -invert.matrix[1][0];
    neg_Rtranspose.matrix[1][1] = -invert.matrix[1][1];
    neg_Rtranspose.matrix[1][2] = -invert.matrix[1][2];

    neg_Rtranspose.matrix[2][0] = -invert.matrix[2][0];
    neg_Rtranspose.matrix[2][1] = -invert.matrix[2][1];
    neg_Rtranspose.matrix[2][2] = -invert.matrix[2][2];

    float3 vec;
    vec.x = a.matrix[0][3];
    vec.y = a.matrix[1][3];
    vec.z = a.matrix[2][3];

    float3 t = mat3_mul_v3(neg_Rtranspose, vec);
    invert.matrix[0][3] = t.x;
    invert.matrix[1][3] = t.y;
    invert.matrix[2][3] = t.z;

    invert.matrix[3][0] = 0.0f;
    invert.matrix[3][1] = 0.0f;
    invert.matrix[3][2] = 0.0f;
    invert.matrix[3][3] = 1.0f;

    return invert;
}


/**************************** geometric operation **************************/

inline mat4 trans(float4 v)
{
    mat4 t = zeroes4x4();

    t.matrix[0][0] = 1.0f;
    t.matrix[1][1] = 1.0f;
    t.matrix[2][2] = 1.0f;
    t.matrix[3][3] = 1.0f;
    t.matrix[0][3] = v.x;
    t.matrix[1][3] = v.y;
    t.matrix[2][3] = v.z;
    return t;
}


inline mat4 rotX(float theta)
{
    mat4 t = zeroes4x4();

    t.matrix[0][0] = 1.0f;
    t.matrix[1][1] = cos(theta);
    t.matrix[2][1] = sin(theta);
    t.matrix[1][2] = -1.0f * t.matrix[2][1];
    t.matrix[2][2] = t.matrix[1][1];
    t.matrix[3][3] = 1.0f;

    return t;
}


inline mat4 rotY(float theta)
{
    mat4 t = zeroes4x4();

    t.matrix[0][0] = cos(theta);
    t.matrix[0][2] = sin(theta);
    t.matrix[1][1] = 1.0f;
    t.matrix[2][0] = -1.0f * t.matrix[0][2];
    t.matrix[2][2] = t.matrix[0][0];
    t.matrix[3][3] = 1.0f;

    return t;    
}



inline mat4 rotZ(float theta)
{
    mat4 t = zeroes4x4();

    t.matrix[0][0] = cos(theta);
    t.matrix[1][0] = sin(theta);
    t.matrix[0][1] = -1.0f * t.matrix[1][0];
    t.matrix[1][1] = t.matrix[0][0];
    t.matrix[2][2] = 1.0f;
    t.matrix[3][3] = 1.0f;

    return t;    
}

// https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

inline float4 hrotmat2quat(mat4 t)
{
    float tr, r;
    float4 q;
    tr = t.matrix[0][0] + t.matrix[1][1] + t.matrix[2][2];
    r = sqrt(1.0f + tr);
    q.x = 0.5f * r;
    q.y = copysign(0.5f * sqrt(1.0f + t.matrix[0][0] - t.matrix[1][1] - t.matrix[2][2]), t.matrix[2][1] - t.matrix[1][2]);
    q.z = copysign(0.5f * sqrt(1.0f - t.matrix[0][0] + t.matrix[1][1] - t.matrix[2][2]), t.matrix[0][2] - t.matrix[2][0]);
    q.w = copysign(0.5f * sqrt(1.0f - t.matrix[0][0] - t.matrix[1][1] + t.matrix[2][2]), t.matrix[1][0] - t.matrix[0][1]);
    float n = sqrt(length(q));
    q.x /= n;
    q.y /= n;
    q.z /= n;
    q.w /= n;
    return q;

}

inline float quant_downf(float x, float y)
{
    return x - fmod(x, y);
}
