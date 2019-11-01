// Copyright(c) 2019-present, Gavin Dunlap
// Distributed under the MIT License (http://opensource.org/licenses/MIT)

#ifndef SIECH_VECTOR3D_HPP
#define SIECH_VECTOR3D_HPP

#include <algorithm>
#include <cmath>
#include <immintrin.h> //AVX


class alignas(32) Vector3d
{
public:
	inline Vector3d() noexcept 
		: mmval(_mm256_setzero_pd())  { } 
	inline Vector3d(double x, double y, double z) noexcept 
		: mmval(_mm256_set_pd(0, z, y, x)) { }
	inline Vector3d(__m256d data) noexcept
		: mmval(data) {}
	inline Vector3d(const Vector3d& o) noexcept
		: mmval(o.mmval) {}
	inline Vector3d(Vector3d&& o) noexcept
		: mmval(o.mmval) {}

	inline Vector3d& operator=(const Vector3d& o) noexcept { mmval = o.mmval; }
	inline Vector3d& operator=(Vector3d&& o) noexcept { mmval = o.mmval; }

	inline Vector3d operator+(Vector3d b) const noexcept { return b += *this; }
	inline Vector3d operator-(Vector3d b) const noexcept { return b -= *this; }

	inline Vector3d& operator+=(const Vector3d& b) noexcept { mmval = _mm256_add_pd(mmval, b.mmval); return *this; }
	inline Vector3d& operator-=(const Vector3d& b) noexcept { mmval = _mm256_sub_pd(mmval, b.mmval); return *this; }
	
	inline Vector3d operator+(const double b) const noexcept { return _mm256_add_pd(mmval, _mm256_set1_pd(b)); }
	inline Vector3d operator-(const double b) const noexcept { return _mm256_sub_pd(mmval, _mm256_set1_pd(b)); }
	inline Vector3d operator*(const double b) const noexcept { return _mm256_mul_pd(mmval, _mm256_set1_pd(b)); }
	inline Vector3d operator/(const double b) const noexcept { return _mm256_div_pd(mmval, _mm256_set1_pd(b)); }

	inline Vector3d& operator+=(const double b) noexcept { mmval = _mm256_add_pd(mmval, _mm256_set1_pd(b)); return *this; }
	inline Vector3d& operator-=(const double b) noexcept { mmval = _mm256_sub_pd(mmval, _mm256_set1_pd(b)); return *this; }
	inline Vector3d& operator*=(const double b) noexcept { mmval = _mm256_mul_pd(mmval, _mm256_set1_pd(b)); return *this; }
	inline Vector3d& operator/=(const double b) noexcept { mmval = _mm256_div_pd(mmval, _mm256_set1_pd(b)); return *this; }

	inline Vector3d cross(const Vector3d b) const noexcept
	{	
		//Pre condition: {N/A, a3, a2, a1}(this->mmval), {N/A, b3, b2, b1}(b.mmval)
		//Post condition: {N/A, a1b2 - a2b1, a3b1 - a1b3, a2b3 - a3b2}
		const __m256d shuffle1 = _mm256_shuffle_pd(mmval, mmval, _MM_SHUFFLE(3, 0, 2, 1));		// { N/A, a1, a3, a2 }
		const __m256d shuffle2 = _mm256_shuffle_pd(b.mmval, b.mmval, _MM_SHUFFLE(3, 1, 0, 2));	// { N/A, b2, b1, b3 }
		const __m256d mul1 = _mm256_mul_pd(shuffle1, shuffle2);									// { N/A, a1b2, a3b1, a2b3 }

		const __m256d shuffle3 = _mm256_shuffle_pd(mmval, mmval, _MM_SHUFFLE(3, 1, 0, 2));		// { N/A, a2, a1, a3 }
		const __m256d shuffle4 = _mm256_shuffle_pd(b.mmval, b.mmval, _MM_SHUFFLE(3, 0, 2, 1));	// { N/A, b1, b3, b2 }
		const __m256d mul2 = _mm256_mul_pd(shuffle3, shuffle4);									// { N/A, a2b1, a1b3, a3b2 }

		return _mm256_sub_pd(mul1, mul2);														// { N/A, a1b2 - a2b1, a3b1 - a1b3, a2b3 - a3b2 }
		//What a doozy to look at. 
	}
																				 
	inline double dot(const Vector3d& b) const noexcept 
	{ 
		//Pre condition: {0, a3, a2, a1}(this->mmval), {0, b3, b2, b1}(b.mmval)
		//Post condition: { a3b3 + a2b2 + a1b1, a3b3 + a2b2 + a1b1 }
		const __m256d xy = _mm256_mul_pd(mmval, b.mmval);		// { 0, a3b3, a2b2, a1b1 }
		const __m256d temp = _mm256_hadd_pd(xy, xy);			// { a3b3, a3b3, a2b2 + a1b1, a2b2 + a1b1 }
		const __m128d lo128 = _mm256_extractf128_pd(temp, 0);	// { a3b3, a3b3 }
		const __m128d hi128 = _mm256_extractf128_pd(temp, 1);	// { a2b2 + a1b1, a2b2 + a1b1 }
		const __m128d dot = _mm_add_pd(lo128, hi128);			// { a3b3 + a2b2 + a1b1, a3b3 + a2b2 + a1b1 }
		return _mm_cvtsd_f64(dot);								// Return first.
		//In hindsight, this might as well have been done via the scalar parts.
	}
	
	inline double length() const noexcept 
	{	//Pre condition: {N/A, a3, a2, a1}(this->mmval), {N/A, b3, b2, b1}(b.mmval)	sqrt(a^2 + b^2 + c^2)
		//Post condition: sqrt(a3a3 + a2a2 + a1a1)
		const __m256d sqred = _mm256_mul_pd(mmval, mmval);		// { N/A, a3a3, a2a2, a1a1 }
		const __m256d temp = _mm256_hadd_pd(sqred, sqred);		// { a3a3, a3a3, a2a2+a1a1}
		const __m128d lo128 = _mm256_extractf128_pd(temp, 0);	// { a3a3, a3a3 }
		const __m128d hi128 = _mm256_extractf128_pd(temp, 1);	// { a2a2 + a1a1, a2a2 + a1a1 }
		const __m128d add = _mm_add_pd(lo128, hi128);			// { a3a3 + a2a2 + a1a1, a3a3 + a2a2 + a1a1 }
		const __m128d sqrt = _mm_sqrt_pd(add);					// { sqrt(a3a3 + a2a2 + a1a1, a3a3 + a2a2 + a1a1 }
		return _mm_cvtsd_f64(sqrt);				
	}
	double lengthSquared() const noexcept 
	{ 	//Pre condition: {0, a3, a2, a1}(this->mmval), {0, b3, b2, b1}(b.mmval)
		//Post condition:	a3a3 + a2a2 + a1a1
		const __m256d sqred = _mm256_mul_pd(mmval, mmval);		// { 0, a3a3, a2a2, a1a1 }
		const __m256d temp = _mm256_hadd_pd(sqred, sqred);		// { a3a3, a3a3, a2a2+a1a1, a2a2+a1a1 }
		const __m128d lo128 = _mm256_extractf128_pd(temp, 0);	// { a3a3, a3a3 }
		const __m128d hi128 = _mm256_extractf128_pd(temp, 1);	// { a2a2 + a1a1, a2a2 + a1a1 }
		const __m128d add = _mm_add_pd(lo128, hi128);			// { a3a3 + a2a2 + a1a1, a3a3 + a2a2 + a1a1 }
		return _mm_cvtsd_f64(add);
	}
	
	inline Vector3d normalize() const noexcept
	{ 
		const double mag = this->length();
		return _mm256_div_pd(mmval, _mm256_set1_pd(mag));
	}

	inline Vector3d lerp(const Vector3d& b, double percent) const noexcept { return _mm256_add_pd(mmval, _mm256_mul_pd(_mm256_set1_pd(percent), _mm256_sub_pd(b.mmval, mmval)));}
	inline Vector3d Nlerp(const Vector3d& b, double percent) {return this->lerp(b, percent).normalize();}
	inline Vector3d slerp(const Vector3d& b, double percent) const noexcept
	{
		const double dot = this->dot(b);
		std::clamp(dot, -1.0, 1.0);
		const double theta = std::acos(dot) * percent;
		Vector3d	 relVec = Vector3d(_mm256_sub_pd(b.mmval, _mm256_mul_pd(mmval, _mm256_set1_pd(percent)))); 
		relVec.normalize();
		return _mm256_add_pd(_mm256_mul_pd(mmval, _mm256_set1_pd(std::cos(theta))), _mm256_mul_pd(relVec.mmval, _mm256_set1_pd(theta)));
	}

	inline void *operator new(size_t size) { return ::operator new(size, (std::align_val_t)32); }
	inline void *operator new[](size_t size) { return ::operator new(size, (std::align_val_t)32); }
	inline void operator delete(void *o) { if (o) ::operator delete(o); }
	inline void operator delete[](void* o) { if (o) ::operator delete(o); }

	inline friend Vector3d operator+(double a, const Vector3d& b) noexcept { return _mm256_add_pd(b.mmval, _mm256_set1_pd(a)); }
	inline friend Vector3d operator-(double a, const Vector3d& b) noexcept { return _mm256_sub_pd(b.mmval, _mm256_set1_pd(a)); }
	inline friend Vector3d operator*(double a, const Vector3d& b) noexcept { return _mm256_mul_pd(b.mmval, _mm256_set1_pd(a)); }
	inline friend Vector3d operator/(double a, const Vector3d& b) noexcept { return _mm256_div_pd(b.mmval, _mm256_set1_pd(a)); }
private:
	union
	{
		struct { double x, y, z; };
		__m256d mmval;
	};
};

#endif