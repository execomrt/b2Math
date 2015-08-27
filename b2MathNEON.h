/*
 * Copyright (c) 2006-2009 Erin Catto http://www.box2d.org
 * Copyright (c) 2015 http://www.v3x.net (SIMD Port)
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef B2_MATH_SIMD_H
#define B2_MATH_SIMD_H


#include <arm_neon.h>

typedef float32x2_t v2sf;  // vector of 2 float
typedef float32x4_t v4sf;  // vector of 4 float
typedef uint32x4_t v4su;  // vector of 4 uint32
typedef int32x4_t v4si;  // vector of 4 uint32

#define __m128 float32x4_t

SIMD_FORCE_INLINE float32x4_t _mm_sub_ps(float32x4_t a, float32x4_t b) { return vsubq_f32(a, b); }
SIMD_FORCE_INLINE float32x2_t _mm_sub_ps(float32x2_t a, float32x2_t b) { return vsub_f32(a, b); }
SIMD_FORCE_INLINE float32x4_t _mm_add_ps(float32x4_t a, float32x4_t b) { return vaddq_f32(a, b); }
SIMD_FORCE_INLINE float32x2_t _mm_add_ps(float32x2_t a, float32x2_t b) { return vadd_f32(a, b); }
SIMD_FORCE_INLINE float32x4_t _mm_mul_ps(float32x4_t a, float32x4_t b) { return vmulq_f32(a, b); }
SIMD_FORCE_INLINE float32x2_t _mm_mul_ps(float32x2_t a, float32x2_t b) { return vmul_f32(a, b); }
SIMD_FORCE_INLINE float32x4_t _mm_min_ps(float32x4_t a, float32x4_t b) { return vminq_f32(a, b); }
SIMD_FORCE_INLINE float32x2_t _mm_min_ps(float32x2_t a, float32x2_t b) { return vmin_f32(a, b); }
SIMD_FORCE_INLINE float32x4_t _mm_max_ps(float32x4_t a, float32x4_t b) { return vmaxq_f32(a, b); }
SIMD_FORCE_INLINE float32x2_t _mm_max_ps(float32x2_t a, float32x2_t b) { return vmax_f32(a, b); }

#ifdef __ARM_FEATURE_FMA

/// _mm_fmadd_ps: a * b + c
/// vfma -> Vc[i] += Va[i] + Vb[i]

SIMD_FORCE_INLINE float32x2_t _mm_fmadd_ps(float32x2_t a, float32x2_t b, float32x2_t c) { return vfma_f32(c, b, a); }
SIMD_FORCE_INLINE float32x4_t _mm_fmadd_ps(float32x4_t a, float32x4_t b, float32x4_t c) { return vfmaq_f32(c, b, a); }
SIMD_FORCE_INLINE float32x2_t _mm_fmadd_ps(float32 a, float32x2_t b, float32x2_t c) { return vfma_f32(c, b, vdup_n_f32(a)); }
SIMD_FORCE_INLINE float32x4_t _mm_fmadd_ps(float32 a, float32x4_t b, float32x4_t c) { return vfmaq_f32(c, b, vdupq_n_f32(a)); }

SIMD_FORCE_INLINE float32x2_t _mm_fmsub_ps(float32x2_t a, float32x2_t b, float32x2_t c) { return vfms_f32(c, b, a); }
SIMD_FORCE_INLINE float32x4_t _mm_fmsub_ps(float32x4_t a, float32x4_t b, float32x4_t c) { return vfmsq_f32(c, b, a); }
SIMD_FORCE_INLINE float32x2_t _mm_fmsub_ps(float32 a, float32x2_t b, float32x2_t c) { return vfms_f32(c, b, vdup_n_f32(a)); }
SIMD_FORCE_INLINE float32x4_t _mm_fmsub_ps(float32 a, float32x4_t b, float32x4_t c) { return vfmsq_f32(c, b, vdupq_n_f32(a)); }


#else

 /// vmla -> Vr[i] := Va[i] + Vb[i] * Vc[i]
SIMD_FORCE_INLINE float32x2_t _mm_fmadd_ps(float32x2_t a, float32x2_t b, float32x2_t c) { return vmla_f32(c, b, a); }
SIMD_FORCE_INLINE float32x4_t _mm_fmadd_ps(float32x4_t a, float32x4_t b, float32x4_t c) { return vmlaq_f32(c, b, a); }
SIMD_FORCE_INLINE float32x2_t _mm_fmadd_ps(float32 a, float32x2_t b, float32x2_t c) { return vmla_n_f32(c, b, a); }
SIMD_FORCE_INLINE float32x4_t _mm_fmadd_ps(float32 a, float32x4_t b, float32x4_t c) { return vmlaq_n_f32(c, b, a); }
SIMD_FORCE_INLINE float32x2_t _mm_fmsub_ps(float32x2_t a, float32x2_t b, float32x2_t c) { return vmls_f32(c, b, a); }
SIMD_FORCE_INLINE float32x4_t _mm_fmsub_ps(float32x4_t a, float32x4_t b, float32x4_t c) { return vmlsq_f32(c, b, a); }
SIMD_FORCE_INLINE float32x2_t _mm_fmsub_ps(float32 a, float32x2_t b, float32x2_t c) { return vmls_n_f32(c, b, a); }
SIMD_FORCE_INLINE float32x4_t _mm_fmsub_ps(float32 a, float32x4_t b, float32x4_t c) { return vmlsq_n_f32(c, b, a); }

#endif // __ARM_FEATURE_FMA

SIMD_FORCE_INLINE float32x4_t _mm_set_ps(float _w, float _z, float _y, float _x)
{
    float32x4_t ret;
    ret = vsetq_lane_f32(_x, ret, 0);
    ret = vsetq_lane_f32(_y, ret, 1);
    ret = vsetq_lane_f32(_z, ret, 2);
    ret = vsetq_lane_f32(_w, ret, 3);
    return ret;
}

#define _mm_set_ps1(value) vdup_n_f32(value)

SIMD_FORCE_INLINE float b2ExtractPS1(__m128 vec)
{
    return vgetq_lane_f32(vec, 0);
}

/// This function is used to ensure that a floating point number is not a NaN or infinity.
SIMD_FORCE_INLINE bool b2IsValid(float32 x)
{
    int32 ix = *reinterpret_cast<int32*>(&x);
    return (ix & 0x7f800000) != 0x7f800000;
}

/// This is a approximate yet fast inverse square-root.
SIMD_FORCE_INLINE float32 b2InvSqrt(float32 x)
{
    union
    {
        float32 x;
        int32 i;
    } convert;
    
    convert.x = x;
    float32 xhalf = 0.5f * x;
    convert.i = 0x5f3759df - (convert.i >> 1);
    x = convert.x;
    x = x * (1.5f - xhalf * x * x);
    return x;
}

#define	b2Sqrt(x)	sqrtf(x)
#define	b2Atan2(y, x)	atan2f(y, x)

/// A 2D column vector.
ALIGN16_BEG
struct b2Vec2
{
    /// Default constructor does nothing (for performance).
    b2Vec2() {}
    
    explicit SIMD_FORCE_INLINE b2Vec2(v2sf vec_) : vec(vec_) {}
    
    /// Construct using coordinates.
    SIMD_FORCE_INLINE b2Vec2(float32 x_, float32 y_)  { x = x_; y = y_; }
    
    /// Set this vector to all zeros.
    SIMD_FORCE_INLINE void SetZero() { vec = vdup_n_f32(0); }
    
    /// Set this vector to some specified coordinates.
    SIMD_FORCE_INLINE void Set(float32 x_, float32 y_) { x = x_; y = y_; }
    
    /// Negate this vector.
    SIMD_FORCE_INLINE b2Vec2 operator -() const { b2Vec2 v; v.vec = _mm_sub_ps(vdup_n_f32(0), vec); return v; }
    
    
    /// Read from and indexed element.
    float32 operator () (int32 i) const
    {
        return (&x)[i];
    }
    
    /// Write to an indexed element.
    float32& operator () (int32 i)
    {
        return (&x)[i];
    }
   
    // Test if any lane is strict positive (see b2TestOverlap)
    SIMD_FORCE_INLINE bool HasStrictPositiveLane() const
    {
        uint32x2_t vCmp = vcgt_f32(vec, vdup_n_f32(0));
        return vget_lane_u32(vCmp, 0) || vget_lane_u32(vCmp, 1);
    }
    
    
    /// Add a vector to this vector.
    SIMD_FORCE_INLINE void operator += (const b2Vec2& v)
    {
        vec = _mm_add_ps(vec, v.vec);
    }
    
    /// Subtract a vector from this vector.
    SIMD_FORCE_INLINE void operator -= (const b2Vec2& v)
    {
        vec = _mm_sub_ps(vec, v.vec);
    }
    
    /// Multiply this vector by a scalar.
    SIMD_FORCE_INLINE void operator *= (float32 s)
    {
        vec = _mm_mul_ps(vec, vdup_n_f32(s));
    }
    
    /// Get the length of this vector (the norm).
    SIMD_FORCE_INLINE float32 Length() const
    {
        return b2Sqrt(LengthSquared());
    }
    
    /// Get the length squared. For performance, use this instead of
    /// b2Vec2::Length (if possible).
    SIMD_FORCE_INLINE float32 LengthSquared() const
    {
        float32x2_t v = vmul_f32(vec, vec);
        v = vpadd_f32(v, v);
        return vget_lane_f32(v, 0);
    }
    
    /// Convert this vector into a unit vector. Returns the length.
    float32 Normalize()
    {
        float32 length = Length();
        if (length < b2_epsilon)
        {
            return 0.0f;
        }
        float32 invLength = 1.0f / length;
        vec = _mm_mul_ps(vec, vdup_n_f32(invLength));
        
        return length;
    }
    
    /// Does this vector contain finite coordinates?
    bool IsValid() const
    {
        return b2IsValid(x) && b2IsValid(y);
    }
    
    /// Get the skew vector such that dot(skew_vec, other) == cross(vec, other)
    b2Vec2 Skew() const
    {
        return b2Vec2(-y, x);
    }
    
    
    union
    {
        v2sf vec;
        struct
        {
            float32 x, y;
        };
        
    };
    
}
ALIGN16_END;

/// A 2D column vector with 3 elements.
ALIGN16_BEG
struct b2Vec3
{
    /// Default constructor does nothing (for performance).
    b2Vec3() {}
    
    explicit SIMD_FORCE_INLINE b2Vec3(v4sf vec_) : vec(vec_) {}
    
    /// Construct using coordinates.
    b2Vec3(float32 x_, float32 y_, float32 z_)  { vec = _mm_set_ps(0, z_, y_, x_); }
    
    /// Set this vector to all zeros.
    void SetZero() { vec = vdupq_n_f32(0); }
    
    /// Set this vector to some specified coordinates.
    void Set(float32 x_, float32 y_, float32 z_) { vec = _mm_set_ps(0, z_, y_, x_); }
    
    /// Negate this vector.
    b2Vec3 operator -() const { b2Vec3 v; v.vec = _mm_sub_ps(vdupq_n_f32(0), vec); return v; }
    
    /// Add a vector to this vector.
    SIMD_FORCE_INLINE void operator += (const b2Vec3& v)
    {
        vec = _mm_add_ps(vec, v.vec);
    }
    
    /// Subtract a vector from this vector.
    SIMD_FORCE_INLINE void operator -= (const b2Vec3& v)
    {
        vec = _mm_sub_ps(vec, v.vec);
    }
    
    /// Multiply this vector by a scalar.
    SIMD_FORCE_INLINE void operator *= (float32 s)
    {
        vec = _mm_mul_ps(vec, vdupq_n_f32(s));
    }
    
    union
    {
        v4sf vec;
        struct
        {
            float32 x, y, z;
        };
        
    };
}ALIGN16_END;

/// A 2-by-2 matrix. Stored in column-major order.
struct b2Mat22
{
    /// The default constructor does nothing (for performance).
    b2Mat22() {}
    
    /// Construct this matrix using columns.
    b2Mat22(const b2Vec2& c1, const b2Vec2& c2)
    {
        ex = c1;
        ey = c2;
    }
    
    /// Construct this matrix using scalars.
    b2Mat22(float32 a11, float32 a12, float32 a21, float32 a22)
    {
        ex.x = a11; ex.y = a21;
        ey.x = a12; ey.y = a22;
    }
    
    /// Initialize this matrix using columns.
    void Set(const b2Vec2& c1, const b2Vec2& c2)
    {
        ex = c1;
        ey = c2;
    }
    
    /// Set this to the identity matrix.
    void SetIdentity()
    {
        ex.x = 1.0f; ey.x = 0.0f;
        ex.y = 0.0f; ey.y = 1.0f;
    }
    
    /// Set this matrix to all zeros.
    void SetZero()
    {
        ex.x = 0.0f; ey.x = 0.0f;
        ex.y = 0.0f; ey.y = 0.0f;
    }
    
    b2Mat22 GetInverse() const
    {
        float32 a = ex.x, b = ey.x, c = ex.y, d = ey.y;
        b2Mat22 B;
        float32 det = a * d - b * c;
        if (det != 0.0f)
        {
            det = 1.0f / det;
        }
        B.ex.x = det * d;	B.ey.x = -det * b;
        B.ex.y = -det * c;	B.ey.y = det * a;
        return B;
    }
    
    /// Solve A * x = b, where b is a column vector. This is more efficient
    /// than computing the inverse in one-shot cases.
    b2Vec2 Solve(const b2Vec2& b) const
    {
        float32 a11 = ex.x, a12 = ey.x, a21 = ex.y, a22 = ey.y;
        float32 det = a11 * a22 - a12 * a21;
        if (det != 0.0f)
        {
            det = 1.0f / det;
        }
        b2Vec2 x;
        x.x = det * (a22 * b.x - a12 * b.y);
        x.y = det * (a11 * b.y - a21 * b.x);
        return x;
    }
    
    b2Vec2 ex, ey;
};

/// A 3-by-3 matrix. Stored in column-major order.
struct b2Mat33
{
    /// The default constructor does nothing (for performance).
    b2Mat33() {}
    
    /// Construct this matrix using columns.
    b2Mat33(const b2Vec3& c1, const b2Vec3& c2, const b2Vec3& c3)
    {
        ex = c1;
        ey = c2;
        ez = c3;
    }
    
    /// Set this matrix to all zeros.
    void SetZero()
    {
        ex.SetZero();
        ey.SetZero();
        ez.SetZero();
    }
    
    /// Solve A * x = b, where b is a column vector. This is more efficient
    /// than computing the inverse in one-shot cases.
    b2Vec3 Solve33(const b2Vec3& b) const;
    
    /// Solve A * x = b, where b is a column vector. This is more efficient
    /// than computing the inverse in one-shot cases. Solve only the upper
    /// 2-by-2 matrix equation.
    b2Vec2 Solve22(const b2Vec2& b) const;
    
    /// Get the inverse of this matrix as a 2-by-2.
    /// Returns the zero matrix if singular.
    void GetInverse22(b2Mat33* M) const;
    
    /// Get the symmetric inverse of this matrix as a 3-by-3.
    /// Returns the zero matrix if singular.
    void GetSymInverse33(b2Mat33* M) const;
    
    b2Vec3 ex, ey, ez;
};

/// Rotation
struct b2Rot
{
    b2Rot() {}
    
    /// Initialize from an angle in radians
    explicit b2Rot(float32 angle)
    {
        Set(angle);
    }
    
    /// Set using an angle in radians. Precision is very critical
    inline void Set(float32 _val)
    {
#if defined __APPLE__ && (__IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_7_0 || __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_9)
		__sincosf(_val, &s, &c); 
#elif defined(_GNU_SOURCE_NOT_WORKING)
        __builtin_sincosf(_val, s, c); // NDK API LEVEL 9. FIXME: Link Error
#else
   	  s = sinf(_val);
  	  c = cosf(_val);
#endif        
    }
    
    /// Set to the identity rotation
    void SetIdentity()
    {
        s = 0.0f;
        c = 1.0f;        
    }
    
    /// Get the angle in radians
    float32 GetAngle() const
    {
        return b2Atan2(s, c);
    }
    
    /// Get the x-axis
    b2Vec2 GetXAxis() const
    {
        return b2Vec2(c, s);
    }
    
    /// Get the u-axis
    b2Vec2 GetYAxis() const
    {
        return b2Vec2(-s, c);
    }
    
    float32 c,s;
 
};


#define _PS_CONST(Name, Val)                                            \
static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { (float)Val, (float)Val, (float)Val, (float)Val }
#define _PI32_CONST(Name, Val)                                            \
static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }

/// A transform contains translation and rotation. It is used to represent
/// the position and orientation of rigid frames.
struct b2Transform
{
    /// The default constructor does nothing.
    b2Transform() {}
    
    /// Initialize using a position vector and a rotation.
    b2Transform(const b2Vec2& position, const b2Rot& rotation) : p(position), q(rotation) {}
    
    /// Set  this to the identity transform.
    void SetIdentity()
    {
        p.SetZero();
        q.SetIdentity();
    }
    
    /// Set this based on the position and angle.
    void Set(const b2Vec2& position, float32 angle)
    {
        p = position;
        q.Set(angle);
    }
    
    b2Vec2 p;
    b2Rot q;
};

/// This describes the motion of a body/shape for TOI computation.
/// Shapes are defined with respect to the body origin, which may
/// no coincide with the center of mass. However, to support dynamics
/// we must interpolate the center of mass position.
struct b2Sweep
{
    /// Get the interpolated transform at a specific time.
    /// @param beta is a factor in [0,1], where 0 indicates alpha0.
    void GetTransform(b2Transform* xfb, float32 beta) const;
    
    /// Advance the sweep forward, yielding a new initial state.
    /// @param alpha the new initial time.
    void Advance(float32 alpha);
    
    /// Normalize the angles.
    void Normalize();
    
    b2Vec2 localCenter;	///< local center of mass position
    b2Vec2 c0, c;		///< center world positions
    float32 a0, a;		///< world angles
    
    /// Fraction of the current time step in the range [0,1]
    /// c0 and a0 are the positions at alpha0.
    float32 alpha0;
};

/// Useful constant
extern const b2Vec2 b2Vec2_zero;

/// Perform the dot product on two vectors.
SIMD_FORCE_INLINE float32 b2Dot(const b2Vec2& a, const b2Vec2& b)
{
    float32x2_t v = vmul_f32(a.vec, b.vec);
    v = vpadd_f32(v, v);
    return vget_lane_f32(v, 0);
}

/// Fused multiply-add. Box2D code was modified to use this macro
SIMD_FORCE_INLINE b2Vec2 b2Madd(const b2Vec2& t, const b2Vec2& v, const float k)
{
    return b2Vec2(_mm_fmadd_ps(k, v.vec, t.vec)); // b2Vec2(t.x + v.x * k, t.y + v.y * k)
}

SIMD_FORCE_INLINE b2Vec2 b2Msub(const b2Vec2& t, const b2Vec2& v, const float k)
{
    return b2Vec2(_mm_fmsub_ps(k, v.vec, t.vec)); // b2Vec2(t.x + v.x * k, t.y + v.y * k)
}

/// Perform the cross product on two vectors. In 2D this produces a scalar.
SIMD_FORCE_INLINE float32 b2Cross(const b2Vec2& a, const b2Vec2& b)
{
    return a.x * b.y - a.y * b.x;
}

/// Perform the cross product on a vector and a scalar. In 2D this produces
/// a vector.
SIMD_FORCE_INLINE b2Vec2 b2Cross(const b2Vec2& a, float32 s)
{
    return b2Vec2(s * a.y, -s * a.x);
}

/// Perform the cross product on a scalar and a vector. In 2D this produces
/// a vector.
SIMD_FORCE_INLINE b2Vec2 b2Cross(float32 s, const b2Vec2& a)
{
    return b2Vec2(-s * a.y, s * a.x);
}

/// Multiply a matrix times a vector. If a rotation matrix is provided,
/// then this transforms the vector from one frame to another.
SIMD_FORCE_INLINE b2Vec2 b2Mul(const b2Mat22& A, const b2Vec2& v)
{
    // return b2Vec2(A.ex.x * v.x + A.ey.x * v.y, A.ex.y * v.x + A.ey.y * v.y);
    v2sf ret =  _mm_mul_ps(vdup_n_f32(v.x), A.ex.vec);
    ret = _mm_fmadd_ps(vdup_n_f32(v.y), A.ey.vec, ret); // _mm_fmadd_ps is ideal in this case
    return b2Vec2(ret);
}

/// Multiply a matrix transpose times a vector. If a rotation matrix is provided,
/// then this transforms the vector from one frame to another (inverse transform).
SIMD_FORCE_INLINE b2Vec2 b2MulT(const b2Mat22& A, const b2Vec2& v)
{
    return b2Vec2(b2Dot(v, A.ex), b2Dot(v, A.ey));
}

/// Add two vectors component-wise.
SIMD_FORCE_INLINE b2Vec2 operator + (const b2Vec2& a, const b2Vec2& b)
{
    return b2Vec2(_mm_add_ps(a.vec, b.vec));   
}

/// Subtract two vectors component-wise.
SIMD_FORCE_INLINE b2Vec2 operator - (const b2Vec2& a, const b2Vec2& b)
{
    return b2Vec2(_mm_sub_ps(a.vec, b.vec));
}

SIMD_FORCE_INLINE b2Vec2 operator * (float32 s, const b2Vec2& a)
{
    return b2Vec2(_mm_mul_ps(a.vec, vdup_n_f32(s)));
}

SIMD_FORCE_INLINE bool operator == (const b2Vec2& a, const b2Vec2& b)
{
    return a.x == b.x && a.y == b.y;
}

SIMD_FORCE_INLINE float32 b2Distance(const b2Vec2& a, const b2Vec2& b)
{
    b2Vec2 c = a - b;
    return c.Length();
}

SIMD_FORCE_INLINE float32 b2DistanceSquared(const b2Vec2& a, const b2Vec2& b)
{
    b2Vec2 c = a - b;
    return b2Dot(c, c);
}

SIMD_FORCE_INLINE b2Vec3 operator * (float32 s, const b2Vec3& a)
{
    return b2Vec3(_mm_mul_ps(a.vec, vdupq_n_f32(s)));
}

/// Add two vectors component-wise.
SIMD_FORCE_INLINE b2Vec3 operator + (const b2Vec3& a, const b2Vec3& b)
{
    return b2Vec3(_mm_add_ps(a.vec, b.vec));
}

/// Subtract two vectors component-wise.
SIMD_FORCE_INLINE b2Vec3 operator - (const b2Vec3& a, const b2Vec3& b)
{
    return b2Vec3(_mm_sub_ps(a.vec, b.vec));
}

/// Perform the dot product on two vectors.
SIMD_FORCE_INLINE float32 b2Dot(const b2Vec3& a, const b2Vec3& b)
{
    // a.w or b.w must be zero (vmul->vmla->vpadd)
   	float32x2_t r = vmla_f32(vmul_f32(
                                      vget_high_f32(a.vec),
                                      vget_high_f32(b.vec)),
                                      vget_low_f32(a.vec),
                                      vget_low_f32(b.vec));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}

/// Perform the cross product on two vectors.
SIMD_FORCE_INLINE b2Vec3 b2Cross(const b2Vec3& a, const b2Vec3& b)
{
    float32x4_t v1 = a.vec;
    float32x4_t v2 = b.vec;
    float32x4x2_t v_1203 = vzipq_f32(vcombine_f32(vrev64_f32(vget_low_f32(v1)), vrev64_f32(vget_low_f32(v2))), vcombine_f32(vget_high_f32(v1), vget_high_f32(v2)));
    float32x4x2_t v_2013 = vzipq_f32(vcombine_f32(vrev64_f32(vget_low_f32(v_1203.val[0])), vrev64_f32(vget_low_f32(v_1203.val[1]))), vcombine_f32(vget_high_f32(v_1203.val[0]), vget_high_f32(v_1203.val[1])));
    return b2Vec3(vmlsq_f32(vmulq_f32(v_1203.val[0], v_2013.val[1]), v_1203.val[1], v_2013.val[0]));
}

SIMD_FORCE_INLINE b2Mat22 operator + (const b2Mat22& A, const b2Mat22& B)
{
    return b2Mat22(A.ex + B.ex, A.ey + B.ey);
}

// A * B
SIMD_FORCE_INLINE b2Mat22 b2Mul(const b2Mat22& A, const b2Mat22& B)
{
    return b2Mat22(b2Mul(A, B.ex), b2Mul(A, B.ey));
}

// A^T * B
SIMD_FORCE_INLINE b2Mat22 b2MulT(const b2Mat22& A, const b2Mat22& B)
{
    b2Vec2 c1(b2Dot(A.ex, B.ex), b2Dot(A.ey, B.ex));
    b2Vec2 c2(b2Dot(A.ex, B.ey), b2Dot(A.ey, B.ey));
    return b2Mat22(c1, c2);
}

/// Multiply a matrix times a vector.
SIMD_FORCE_INLINE b2Vec3 b2Mul(const b2Mat33& A, const b2Vec3& v)
{
    v4sf ret =  _mm_mul_ps(vdupq_n_f32(v.x), A.ex.vec);
    ret = _mm_fmadd_ps(vdupq_n_f32(v.y), A.ey.vec, ret); // _mm_fmadd_ps is ideal in this case
    ret = _mm_fmadd_ps(vdupq_n_f32(v.z), A.ez.vec, ret);
    return b2Vec3(ret);
}

/// Multiply a matrix times a vector.
SIMD_FORCE_INLINE b2Vec2 b2Mul22(const b2Mat33& A, const b2Vec2& v)
{
    v4sf ret =  _mm_mul_ps(vdupq_n_f32(v.x), A.ex.vec);
    ret = _mm_fmadd_ps(vdupq_n_f32(v.y), A.ey.vec, ret); // _mm_fmadd_ps is ideal in this case
    return b2Vec2(vget_low_f32(ret));
}

/// Multiply two rotations: q * r
SIMD_FORCE_INLINE b2Rot b2Mul(const b2Rot& q, const b2Rot& r)
{
    // [qc -qs] * [rc -rs] = [qc*rc-qs*rs -qc*rs-qs*rc]
    // [qs  qc]   [rs  rc]   [qs*rc+qc*rs -qs*rs+qc*rc]
    // s = qs * rc + qc * rs
    // c = qc * rc - qs * rs
    b2Rot qr;
    qr.s = q.s * r.c + q.c * r.s;
    qr.c = q.c * r.c - q.s * r.s;
    return qr;
}

/// Transpose multiply two rotations: qT * r
SIMD_FORCE_INLINE b2Rot b2MulT(const b2Rot& q, const b2Rot& r)
{
    // [ qc qs] * [rc -rs] = [qc*rc+qs*rs -qc*rs+qs*rc]
    // [-qs qc]   [rs  rc]   [-qs*rc+qc*rs qs*rs+qc*rc]
    // s = qc * rs - qs * rc
    // c = qc * rc + qs * rs
    b2Rot qr;
    qr.s = q.c * r.s - q.s * r.c;
    qr.c = q.c * r.c + q.s * r.s;
    return qr;
}

/// Rotate a vector
SIMD_FORCE_INLINE b2Vec2 b2Mul(const b2Rot& q, const b2Vec2& v)
{
    return b2Vec2(q.c * v.x - q.s * v.y, q.s * v.x + q.c * v.y);
}

/// Inverse rotate a vector
SIMD_FORCE_INLINE b2Vec2 b2MulT(const b2Rot& q, const b2Vec2& v)
{
    return b2Vec2(q.c * v.x + q.s * v.y, -q.s * v.x + q.c * v.y);
}

SIMD_FORCE_INLINE b2Vec2 b2Mul(const b2Transform& T, const b2Vec2& v)
{
    
    return b2Vec2((T.q.c * v.x - T.q.s * v.y),
                  (T.q.s * v.x + T.q.c * v.y)) + T.p;
}

SIMD_FORCE_INLINE b2Vec2 b2MulT(const b2Transform& T, const b2Vec2& v)
{
    b2Vec2 p = v - T.p;
    
    return b2Vec2((T.q.c * p.x + T.q.s * p.y), (-T.q.s * p.x + T.q.c * p.y));
}

// v2 = A.q.Rot(B.q.Rot(v1) + B.p) + A.p
//    = (A.q * B.q).Rot(v1) + A.q.Rot(B.p) + A.p
SIMD_FORCE_INLINE b2Transform b2Mul(const b2Transform& A, const b2Transform& B)
{
    b2Transform C;
    C.q = b2Mul(A.q, B.q);
    C.p = b2Mul(A.q, B.p) + A.p;
    return C;
}

// v2 = A.q' * (B.q * v1 + B.p - A.p)
//    = A.q' * B.q * v1 + A.q' * (B.p - A.p)
SIMD_FORCE_INLINE b2Transform b2MulT(const b2Transform& A, const b2Transform& B)
{
    b2Transform C;
    C.q = b2MulT(A.q, B.q);
    C.p = b2MulT(A.q, B.p - A.p);
    return C;
}
template <typename T>
SIMD_FORCE_INLINE T b2Abs(T a)
{
    return a > T(0) ? a : -a;
}

SIMD_FORCE_INLINE b2Vec2 b2Abs(const b2Vec2& a)
{
   return b2Vec2(vabs_f32(a.vec));
}

SIMD_FORCE_INLINE b2Mat22 b2Abs(const b2Mat22& A)
{
    return b2Mat22(b2Abs(A.ex), b2Abs(A.ey));
}

template <typename T>
SIMD_FORCE_INLINE T b2Min(T a, T b)
{
    return a < b ? a : b;
}

SIMD_FORCE_INLINE b2Vec2 b2Min(const b2Vec2& a, const b2Vec2& b)
{
    return b2Vec2(_mm_min_ps(a.vec, b.vec));
}

template <typename T>
SIMD_FORCE_INLINE T b2Max(T a, T b)
{
    return a > b ? a : b;
}

SIMD_FORCE_INLINE b2Vec2 b2Max(const b2Vec2& a, const b2Vec2& b)
{
    return b2Vec2(_mm_max_ps(a.vec, b.vec));
}

template <typename T>
SIMD_FORCE_INLINE T b2Clamp(T a, T low, T high)
{
    return b2Max(low, b2Min(a, high));
}

SIMD_FORCE_INLINE b2Vec2 b2Clamp(const b2Vec2& a, const b2Vec2& low, const b2Vec2& high)
{
    return b2Max(low, b2Min(a, high));
}

template<typename T> SIMD_FORCE_INLINE void b2Swap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

/// "Next Largest Power of 2
/// Given a binary integer value x, the next largest power of 2 can be computed by a SWAR algorithm
/// that recursively "folds" the upper bits into the lower bits. This process yields a bit vector with
/// the same most significant 1 as x, but all 1's below it. Adding 1 to that value yields the next
/// largest power of 2. For a 32-bit value:"
SIMD_FORCE_INLINE uint32 b2NextPowerOfTwo(uint32 x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x + 1;
}

SIMD_FORCE_INLINE bool b2IsPowerOfTwo(uint32 x)
{
    bool result = x > 0 && (x & (x - 1)) == 0;
    return result;
}

SIMD_FORCE_INLINE void b2Sweep::GetTransform(b2Transform* xf, float32 beta) const
{
    xf->p = (1.0f - beta) * c0 + beta * c;
    float32 angle = (1.0f - beta) * a0 + beta * a;
    xf->q.Set(angle);
    
    // Shift to origin
    xf->p -= b2Mul(xf->q, localCenter);
}

SIMD_FORCE_INLINE void b2Sweep::Advance(float32 alpha)
{
    b2Assert(alpha0 < 1.0f);
    float32 beta = (alpha - alpha0) / (1.0f - alpha0);
    c0 += beta * (c - c0);
    a0 += beta * (a - a0);
    alpha0 = alpha;
}

/// Normalize an angle in radians to be between -pi and pi
SIMD_FORCE_INLINE void b2Sweep::Normalize()
{
    const float32 twoPi = 2.0f * b2_pi;
    float32 d = twoPi * floorf(a0 / twoPi);
    a0 -= d;
    a -= d;
}

#endif