#ifndef SIECH_VECTOR3F_HPP
#define SIECH_VECTOR3F_HPP

#include <algorithm>
#include <cmath>

#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2

class alignas(16) Vector3
{
public:
	inline Vector3() noexcept 
		: mmval(_mm_setzero_ps())  { } 
	inline Vector3(float x, float y, float z) noexcept 
		: mmval(_mm_set_ps(0, z, y, x)) { }
	inline Vector3(__m128 data) noexcept
		: mmval(data) {}
	inline Vector3(const Vector3& o) noexcept
		: mmval(o.mmval) {}
	inline Vector3(Vector3&& o) noexcept
		: mmval(o.mmval) {}

	inline Vector3& operator=(const Vector3& o) noexcept { mmval = o.mmval; }
	inline Vector3& operator=(Vector3&& o) noexcept { mmval = o.mmval; }
											  
	inline Vector3 operator+(Vector3 b) const noexcept { return b += *this; }
	inline Vector3 operator-(Vector3 b) const noexcept { return b -= *this; }

	inline Vector3& operator+=(const Vector3& b) noexcept { mmval = _mm_add_ps(mmval, b.mmval); return *this; }
	inline Vector3& operator-=(const Vector3& b) noexcept { mmval = _mm_sub_ps(mmval, b.mmval); return *this; }
	
	inline Vector3 operator+(const float b) const noexcept { return _mm_add_ps(mmval, _mm_set1_ps(b)); }
	inline Vector3 operator-(const float b) const noexcept { return _mm_sub_ps(mmval, _mm_set1_ps(b)); }
	inline Vector3 operator*(const float b) const noexcept { return _mm_mul_ps(mmval, _mm_set1_ps(b)); }
	inline Vector3 operator/(const float b) const noexcept { return _mm_div_ps(mmval, _mm_set1_ps(b)); }

	inline Vector3& operator+=(const float b) noexcept { mmval = _mm_add_ps(mmval, _mm_set1_ps(b)); return *this; }
	inline Vector3& operator-=(const float b) noexcept { mmval = _mm_sub_ps(mmval, _mm_set1_ps(b)); return *this; }
	inline Vector3& operator*=(const float b) noexcept { mmval = _mm_mul_ps(mmval, _mm_set1_ps(b)); return *this; }
	inline Vector3& operator/=(const float b) noexcept { mmval = _mm_div_ps(mmval, _mm_set1_ps(b)); return *this; }

	inline Vector3 cross(const Vector3 b) const noexcept
	{	//More detailed explanation in Vector3d::cross
		return _mm_sub_ps(
			_mm_mul_ps(_mm_shuffle_ps(mmval, mmval, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(b.mmval, b.mmval, _MM_SHUFFLE(3, 1, 0, 2))),
			_mm_mul_ps(_mm_shuffle_ps(mmval, mmval, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(b.mmval, b.mmval, _MM_SHUFFLE(3, 0, 2, 1)))
		);
	}

	inline float dot(const Vector3& b) const noexcept { return _mm_cvtss_f32(_mm_dp_ps(mmval, b.mmval, 0b01110001)); }
	inline float	length() const noexcept { return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(mmval, mmval, 0b01110001)));}
	inline float lengthSquared() const noexcept { return _mm_cvtss_f32(_mm_dp_ps(mmval, mmval, 0b01110001)); }
	inline Vector3 normalize() const noexcept { return _mm_mul_ps(mmval, _mm_rsqrt_ps(_mm_dp_ps(mmval, mmval, 0b01111111))); }

	inline Vector3 lerp(const Vector3& b, float percent) const noexcept { return _mm_add_ps(mmval, _mm_mul_ps(_mm_set1_ps(percent), _mm_sub_ps(b.mmval, mmval)));}
	inline Vector3 Nlerp(const Vector3& b, float percent) {return this->lerp(b, percent).normalize();}
	inline Vector3 slerp(const Vector3& b, float percent) const noexcept
	{
		const float dot = this->dot(b);
		std::clamp(dot, -1.0f, 1.0f);
		const float theta = std::acos(dot) * percent;
		Vector3	 relVec = Vector3(_mm_sub_ps(b.mmval, _mm_mul_ps(mmval, _mm_set1_ps(percent)))); 
		relVec.normalize();
		return _mm_add_ps(_mm_mul_ps(mmval, _mm_set1_ps(std::cos(theta))), _mm_mul_ps(relVec.mmval, _mm_set1_ps(theta)));
	}

	inline void *operator new(size_t size) { return ::operator new(size, (std::align_val_t)16); }	//Ensure 16byte alignment
	inline void *operator new[](size_t size) { return ::operator new(size, (std::align_val_t)16); }
	inline void operator delete(void *o) { if (o) ::operator delete(o); }
	inline void operator delete[](void* o) { if (o) ::operator delete(o); }

	inline friend Vector3 operator+(float a, const Vector3& b) noexcept { return _mm_add_ps(b.mmval, _mm_set1_ps(a)); }
	inline friend Vector3 operator-(float a, const Vector3& b) noexcept { return _mm_sub_ps(b.mmval, _mm_set1_ps(a)); }
	inline friend Vector3 operator*(float a, const Vector3& b) noexcept { return _mm_mul_ps(b.mmval, _mm_set1_ps(a)); }
	inline friend Vector3 operator/(float a, const Vector3& b) noexcept { return _mm_div_ps(b.mmval, _mm_set1_ps(a)); }

private:
	union
	{
		struct { float x, y, z; };
		__m128 mmval;	
	};
};



#endif