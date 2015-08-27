/*
* Copyright (c) 2006-2009 Erin Catto http://www.box2d.org
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

#ifndef B2_MATH_H
#define B2_MATH_H

#include <float.h>
#include <math.h>

#include <Box2D/Common/b2Settings.h>

#ifdef __ARM_NEON__
// ARM Neon
#define B2_TARGET_NEON
// SSE
#elif defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86_FP) || defined __i386__
#if !defined B2_TARGET_SSE
#define B2_TARGET_SSE 3 // SSE3 (Atom - Q1'04)
//#define B2_TARGET_SSE 4 // SSE4 (Pentium 4 - Q2'07) (use of _mm_dp_ps)
//#define B2_TARGET_SSE 5 // FMA/LZN  (4th gen Intel Core - Q2'13) (use of _mm_fmadd_ps)
#endif
#endif

#ifdef B2_TARGET_SSE
#include "b2MathSSE3.h"
#elif defined B2_TARGET_NEON
#include "b2MathNEON.h"
#else
#include "b2MathIEEE.h"
#endif
#endif