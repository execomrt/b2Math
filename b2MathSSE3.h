/*
 * Copyright (c) 2006-2009 Erin Catto http://www.box2d.org
 * Copyright (c) 2015 realtech-VR http://www.v3x.net (SIMD Port)
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

	#define B2_TARGET_SSE 3 // SSE3 (Atom - Q1'04)
	#define B2_TARGET_SSE 4 // SSE4 (Pentium 4 - Q2'07) (use of _mm_dp_ps)
    #define B2_TARGET_SSE 5 // FMA/LZN  (4th gen Intel Core - Q2'13) (use of _mm_fmadd_ps)
 
 */

#ifndef B2_MATH_SIMD_H
#define B2_MATH_SIMD_H

#if _MSC_VER >= 1400
#include <intrin.h>
#else
#include <pmmintrin.h>
#if B2_TARGET_SSE >= 4
// Prescott
#include <smmintrin.h>
#endif
#if B2_TARGET_SSE >= 5
// Haswell
#include <x86intrin.h> // LZCNT
#include <fmaintrin.h>
#endif
#endif

typedef __m128 v4sf;  // vector of 4 float
typedef __m128i v4su;  // vector of 4 uint32
typedef __m128i v4si;  // vector of 4 uint32

// a * b + c
#if B2_TARGET_SSE >= 5
#define b2_fmadd_ps(a, b, c) c = _mm_fmadd_ps(a, b, c)
#define b2_fmsub_ps(a, b, c) c = _mm_fmsub_ps(a, b, c)
#else
#define b2_fmadd_ps(a, b, c) c = _mm_add_ps(_mm_mul_ps(a, b), c)
#define b2_fmsub_ps(a, b, c) c = _mm_sub_ps(_mm_mul_ps(a, b), c)
#endif

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

    explicit SIMD_FORCE_INLINE b2Vec2(v4sf vec_) : vec(vec_) {}

	/// Construct using coordinates.
	SIMD_FORCE_INLINE b2Vec2(float32 x_, float32 y_)  { vec = _mm_set_ps(0, 0, y_, x_); }

	/// Set this vector to all zeros.
	SIMD_FORCE_INLINE void SetZero() { vec = _mm_setzero_ps(); }

	/// Set this vector to some specified coordinates.
	SIMD_FORCE_INLINE void Set(float32 x_, float32 y_) { vec = _mm_set_ps(0, 0, y_, x_); }

	/// Negate this vector.
	SIMD_FORCE_INLINE b2Vec2 operator -() const { b2Vec2 v; v.vec = _mm_sub_ps(_mm_setzero_ps(), vec); return v; }


    // Test if any lane is strict positive (see b2TestOverlap)
    SIMD_FORCE_INLINE bool HasStrictPositiveLane() const
    {
		return x > 0.0f || y > 0.0f;
        // return _mm_movemask_ps(_mm_cmpgt_ps(vec, _mm_setzero_ps())) & (1 | 2) ? true : false;
    }

	/// Read from and indexed element.
	float32 operator () (int32 i) const
	{
		return (&x)[i];
	}

	b2Vec2& operator=(const b2Vec2& other)
	{
		this->vec = other.vec;
		return *this;
	}

	/// Write to an indexed element.
	float32& operator () (int32 i)
	{
		return (&x)[i];
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
		vec = _mm_mul_ps(vec, _mm_load_ps1(&s));
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
#if B2_TARGET_SSE >= 4
		return _mm_cvtss_f32(_mm_dp_ps(vec, vec, 1 | (1 << 4) | (1 << 5))); // Mask 4,5,6,7
#else
		v4sf v2 = _mm_mul_ps(vec, vec);
		return _mm_cvtss_f32(_mm_hadd_ps(v2, v2));
#endif
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
		vec = _mm_mul_ps(vec, _mm_load_ps1(&invLength));

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
		v4sf vec;
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
	void SetZero() { vec = _mm_setzero_ps(); }

	/// Set this vector to some specified coordinates.
	void Set(float32 x_, float32 y_, float32 z_) { vec = _mm_set_ps(0, z_, y_, x_); }

	/// Negate this vector.
	b2Vec3 operator -() const { b2Vec3 v; v.vec = _mm_sub_ps(_mm_setzero_ps(), vec); return v; }

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
		vec = _mm_mul_ps(vec, _mm_load_ps1(&s));
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
    
    /// Set using an angle in radians. Precision is critical
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
    

	/// Initialize from an angle in radians
	explicit b2Rot(float32 angle)
	{		
	    Set(angle);
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

	/// Sine and cosine
    float32 s, c;

	// Cephes SinCos 
	void sincos_ps(v4sf x);
};


#define _PS_CONST(Name, Val)                                            \
static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { (float)Val, (float)Val, (float)Val, (float)Val }
#define _PI32_CONST(Name, Val)                                            \
static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }


#ifdef B2MATH_CPP

/* declare some SSE constants -- why can't I figure a better way to do that? */


_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, (int) 0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, -1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, -1.2420140846E-1);
_PS_CONST(cephes_log_p4, +1.4249322787E-1);
_PS_CONST(cephes_log_p5, -1.6668057665E-1);
_PS_CONST(cephes_log_p6, +2.0000714765E-1);
_PS_CONST(cephes_log_p7, -2.4999993993E-1);
_PS_CONST(cephes_log_p8, +3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);

_PS_CONST(minus_cephes_DP1, -0.78515625);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS_CONST(sincof_p0, -1.9515295891E-4);
_PS_CONST(sincof_p1, 8.3321608736E-3);
_PS_CONST(sincof_p2, -1.6666654611E-1);
_PS_CONST(coscof_p0, 2.443315711809948E-005);
_PS_CONST(coscof_p1, -1.38873162549);
_PS_CONST(coscof_p2, 4.166664568298827E-002);
_PS_CONST(cephes_FOPI, 1.27323954473516); // 4 / b2_pi


void b2Rot::sincos_ps(v4sf x)
{
	// From http://gruntthepeon.free.fr/ssemath/sse_mathfun.h
	/* since sin_ps and cos_ps are almost identical, sincos_ps could replace both of them..
	it is almost as fast, and gives you a free cosine with your sine */
	// 900% faster

	v4sf xmm1, xmm2, xmm3 = _mm_setzero_ps(), sign_bit_sin, y;
	v4si emm0, emm2, emm4;
	sign_bit_sin = x;
	/* take the absolute value */
	x = _mm_and_ps(x, *(const v4sf*) _ps_inv_sign_mask);
	/* extract the sign bit (upper one) */
	sign_bit_sin = _mm_and_ps(sign_bit_sin, *(const v4sf*) _ps_sign_mask);

	/* scale by 4/Pi */
	y = _mm_mul_ps(x, *(const v4sf*) _ps_cephes_FOPI);

	/* store the integer part of y in emm2 */
	emm2 = _mm_cvttps_epi32(y);

	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = _mm_add_epi32(emm2, *(const v4si*) _pi32_1);
	emm2 = _mm_and_si128(emm2, *(const v4si*) _pi32_inv1);
	y = _mm_cvtepi32_ps(emm2);

	emm4 = emm2;

	/* get the swap sign flag for the sine */
	emm0 = _mm_and_si128(emm2, *(const v4si*) _pi32_4);
	emm0 = _mm_slli_epi32(emm0, 29);
	v4sf swap_sign_bit_sin = _mm_castsi128_ps(emm0);

	/* get the polynom selection mask for the sine*/
	emm2 = _mm_and_si128(emm2, *(const v4si*) _pi32_2);
	emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
	v4sf poly_mask = _mm_castsi128_ps(emm2);

	/* The magic pass: "Extended precision modular arithmetic"
	x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v4sf*) _ps_minus_cephes_DP1;
	xmm2 = *(v4sf*) _ps_minus_cephes_DP2;
	xmm3 = *(v4sf*) _ps_minus_cephes_DP3;
	xmm1 = _mm_mul_ps(y, xmm1);
	xmm2 = _mm_mul_ps(y, xmm2);
	xmm3 = _mm_mul_ps(y, xmm3);
	x = _mm_add_ps(x, xmm1);
	x = _mm_add_ps(x, xmm2);
	x = _mm_add_ps(x, xmm3);

	emm4 = _mm_sub_epi32(emm4, *(const v4si*) _pi32_2);
	emm4 = _mm_andnot_si128(emm4, *(const v4si*) _pi32_4);
	emm4 = _mm_slli_epi32(emm4, 29);
	v4sf sign_bit_cos = _mm_castsi128_ps(emm4);

	sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	v4sf z = _mm_mul_ps(x, x);
	y = *(v4sf*) _ps_coscof_p0;

	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(const v4sf*) _ps_coscof_p1);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(const v4sf*) _ps_coscof_p2);
	y = _mm_mul_ps(y, z);
	y = _mm_mul_ps(y, z);
	v4sf tmp = _mm_mul_ps(z, *(const v4sf*) _ps_0p5);
	y = _mm_sub_ps(y, tmp);
	y = _mm_add_ps(y, *(v4sf*) _ps_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

	v4sf y2 = *(v4sf*) _ps_sincof_p0;
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(const v4sf*) _ps_sincof_p1);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(const v4sf*) _ps_sincof_p2);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_mul_ps(y2, x);
	y2 = _mm_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	v4sf ysin2 = _mm_and_ps(xmm3, y2);
	v4sf ysin1 = _mm_andnot_ps(xmm3, y);
	y2 = _mm_sub_ps(y2, ysin2);
	y = _mm_sub_ps(y, ysin1);

	xmm1 = _mm_add_ps(ysin1, ysin2);
	xmm2 = _mm_add_ps(y, y2);

	/* update the sign */
	s = _mm_cvtss_f32(_mm_xor_ps(xmm1, sign_bit_sin));
	c = _mm_cvtss_f32(_mm_xor_ps(xmm2, sign_bit_cos));
}



#endif


/// A transform contains translation and rotation. It is used to represent
/// the position and orientation of rigid frames.
struct b2Transform
{
	/// The default constructor does nothing.
	b2Transform() {}

	/// Initialize using a position vector and a rotation.
	b2Transform(const b2Vec2& position, const b2Rot& rotation) : p(position), q(rotation) {}

	/// Set this to the identity transform.
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
#if B2_TARGET_SSE >= 4
	return _mm_cvtss_f32(_mm_dp_ps(a.vec, b.vec, 1 | (1 << 4) | (1 << 5))); // Mask 4,5,6,7
#else
	v4sf v2 = _mm_mul_ps(a.vec, b.vec);
	return _mm_cvtss_f32(_mm_hadd_ps(v2, v2));
#endif	
}

/// Perform Madd (Mul/Add)
SIMD_FORCE_INLINE b2Vec2 b2Madd(const b2Vec2& t, const b2Vec2& v, const float k)
{    
    auto ret = t.vec;
    b2_fmadd_ps(_mm_load_ps1(&k), v.vec, ret);
	return b2Vec2(ret);
}

SIMD_FORCE_INLINE b2Vec2 b2Msub(const b2Vec2& t, const b2Vec2& v, const float k)
{
	auto ret = t.vec;
    b2_fmsub_ps(_mm_load_ps1(&k), v.vec, ret);
    return b2Vec2(ret);
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
    v4sf ret =  _mm_mul_ps(_mm_set_ps1(v.x), A.ex.vec);
    b2_fmadd_ps(_mm_set_ps1(v.y), A.ey.vec, ret); // _mm_fmadd_ps is ideal in this case
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
	return b2Vec2(_mm_mul_ps(a.vec, _mm_load_ps1(&s)));
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
	return b2Vec3(s * a.x, s * a.y, s * a.z);
}

/// Add two vectors component-wise.
SIMD_FORCE_INLINE b2Vec3 operator + (const b2Vec3& a, const b2Vec3& b)
{
	return b2Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/// Subtract two vectors component-wise.
SIMD_FORCE_INLINE b2Vec3 operator - (const b2Vec3& a, const b2Vec3& b)
{
	return b2Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/// Perform the dot product on two vectors.
SIMD_FORCE_INLINE float32 b2Dot(const b2Vec3& a, const b2Vec3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Perform the cross product on two vectors.
SIMD_FORCE_INLINE b2Vec3 b2Cross(const b2Vec3& a, const b2Vec3& b)
{
	// return b2Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
	
	return b2Vec3(_mm_sub_ps(
    	_mm_mul_ps(_mm_shuffle_ps(a.vec, a.vec, _MM_SHUFFLE(3, 0, 2, 1)), 
	    		   _mm_shuffle_ps(b.vec, b.vec, _MM_SHUFFLE(3, 1, 0, 2))), 
   	 	_mm_mul_ps(_mm_shuffle_ps(a.vec, a.vec, _MM_SHUFFLE(3, 1, 0, 2)), 
          	       _mm_shuffle_ps(b.vec, b.vec, _MM_SHUFFLE(3, 0, 2, 1)))
  	));

	
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
	// return v.x * A.ex + v.y * A.ey + v.z * A.ez;	
    v4sf ret =  _mm_mul_ps(_mm_set_ps1(v.x), A.ex.vec);
    b2_fmadd_ps(_mm_set_ps1(v.y), A.ey.vec, ret); // _mm_fmadd_ps is ideal in this case
    b2_fmadd_ps(_mm_set_ps1(v.z), A.ez.vec, ret);
	return b2Vec3(ret);
}

/// Multiply a matrix times a vector.
SIMD_FORCE_INLINE b2Vec2 b2Mul22(const b2Mat33& A, const b2Vec2& v)
{
	// return b2Vec2(A.ex.x * v.x + A.ey.x * v.y, A.ex.y * v.x + A.ey.y * v.y);
    v4sf ret =  _mm_mul_ps(_mm_set_ps1(v.x), A.ex.vec);
    b2_fmadd_ps(_mm_set_ps1(v.y), A.ey.vec, ret); // _mm_fmadd_ps is ideal in this case
    return b2Vec2(ret);
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
	return b2Vec2((T.q.c * v.x - T.q.s * v.y), (T.q.s * v.x + T.q.c * v.y)) + T.p;
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
	_PS_CONST_TYPE(c7fffffff, int, 0x7fffffff);
	return b2Vec2(_mm_and_ps(a.vec, *(__m128*)&_ps_c7fffffff));
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
/// FIXME: This function is not used by Box2D
SIMD_FORCE_INLINE uint32 b2NextPowerOfTwo(uint32 x)
{
#ifdef __LZCNT__
    int i = 32 - __lzcnt32(x - 1);
    return 1<<i;
#elif defined _MSC_VER >= 1700 && B2_TARGET_SSE >= 5
	int i = 32 - __lzcnt(x - 1); //  LZCNT beginning with the Haswell microarchitecture.
#else
	x |= (x >> 1);
	x |= (x >> 2);
	x |= (x >> 4);
	x |= (x >> 8);
	x |= (x >> 16);
	return x + 1;
#endif
}

/// FIXME: This function is not used by Box2D
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
	float32 twoPi = 2.0f * b2_pi;
	float32 d = twoPi * floorf(a0 / twoPi);
	a0 -= d;
	a -= d;
}

#endif
