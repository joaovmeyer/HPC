#include <iostream>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <x86intrin.h>

#include "../../graph.h"




class Timer {
public:
	Timer() {
		start();
	}

	void start() {
		start_time_ = std::chrono::high_resolution_clock::now();
	}

	void stop() {
		end_time_ = std::chrono::high_resolution_clock::now();
	}

	double elapsedSeconds() const {
		return std::chrono::duration<double>(end_time_ - start_time_).count();
	}

	double elapsedMilliseconds() const {
		return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
	}

	void displayTime() const {
		std::cout << "Elapsed time: " << elapsedMilliseconds() << " ms\n";
	}

private:
	std::chrono::high_resolution_clock::time_point start_time_;
	std::chrono::high_resolution_clock::time_point end_time_;
};





inline float sin_poly_0_pi(float x) {
	return (((0.03681629830044755013571431393746576018286f * x
		- 0.2313236245461128278420738316442693697179f) * x
		+ 0.04891814010265938088474976540346788899839f) * x
		+ 0.9878554618743378113331828267979221403218f) * x;
}


float approx_sin(float x) {
	static constexpr float TWO_PI = 6.28318530717958647692f;
	static constexpr float PI = 3.14159265358979323846f;

	uint32_t sign = std::bit_cast<uint32_t>(x) & 0x80000000;

	x = std::abs(x);
	x -= TWO_PI * static_cast<float>(static_cast<int>(x * (1.0f / TWO_PI)));

	uint32_t big = x >= PI;
	x -= PI * big;

	float s = sin_poly_0_pi(x);

	*reinterpret_cast<uint32_t*>(&s) ^= sign ^ (big << 31);

	return s;
}


// like 30x faster (!!) than std::sin if bulk calculation is needed. Not too accurate
__m256 _mm256_sin_ps(__m256 x) {
	static const __m256 TWO_PI = _mm256_set1_ps(6.28318530717958647692f);
	static const __m256 INV_TWO_PI = _mm256_set1_ps(1.0f / 6.28318530717958647692f);
	static const __m256 PI = _mm256_set1_ps(3.14159265358979323846f);
	static const __m256 sign_bit = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

	__m256 sign = _mm256_and_ps(x, sign_bit); // extract sign bit
	x = _mm256_xor_ps(x, sign); // set sign bit to 0 (abs)

	// range reduction to [0, PI]
	x = _mm256_sub_ps(x, _mm256_mul_ps(TWO_PI, _mm256_floor_ps(_mm256_mul_ps(x, INV_TWO_PI))));
	__m256 big = _mm256_cmp_ps(x, PI, _CMP_GE_OQ);
	x = _mm256_sub_ps(x, _mm256_and_ps(PI, big));

	// evaluate polynomial (Horner's method)
	__m256 s = _mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(
		_mm256_set1_ps(0.03681629830044755013571431393746576018286f), x),
		_mm256_set1_ps(0.23132362454611282784207383164426936971790f)), x),
		_mm256_set1_ps(0.04891814010265938088474976540346788899839f)), x),
		_mm256_set1_ps(0.98785546187433781133318282679792214032180f)), x);

	// adjust the sign (sin(x) = -sin(-x) = -sin(x - PI))
	return _mm256_xor_ps(s, _mm256_xor_ps(sign, _mm256_and_ps(big, sign_bit)));
}





// idea: divide the range [0, pi) into 9 equal parts doing something like this:

// considere x already in the [0, pi) range
// section_limits is the vector with values [pi/9, 2pi/9, ..., 8pi/9]
// __m256 vec_x = _mm256_set1_ps(x);
// uint32_t section_mask = _mm256_movemask_ps(_mm256_cmp_ps(vec_x, section_limits, _CMP_GT_OQ));
// uint32_t section = __builtin_ctz(section_mask); // __builtin_ctz(0) might not work, so maybe do __builtin_ctz((section_mask << 1) | 1) - 1

// also, while current way of reducing x's range to [0, 2pi)] is fast, it's also innacurate. Look into Payne Hanek algorithm


float hsum(__m256 vec) {
	float s = 0.0f;
	for (int i = 0; i < 8; ++i) s += vec[i]; // GCC is smart
	return s;
}

/*
int main() {
	constexpr int N = 1e8;
	float result = 0.0f;
	__m256 results = _mm256_setzero_ps();
	float scaleFactor = 0.00001f;

	Timer timer;

	float next = 0.62456735f;

	// Benchmark approx_sin
	timer.start();
	for (int i = 0; i + 7 < N; i += 8) {
		float x = i * scaleFactor; // approx range [-π, π]
		int a = 0;
		results = _mm256_add_ps(results, _mm256_sin_ps(_mm256_set_ps(x + (a++) * scaleFactor, x + (a++) * scaleFactor, x + (a++) * scaleFactor, x + (a++) * scaleFactor, x + (a++) * scaleFactor, x + (a++) * scaleFactor, x + (a++) * scaleFactor, x + (a++) * scaleFactor)));
	}
	timer.stop();
	double approx_time = timer.elapsedMilliseconds();

	std::cout << "approx_sin next: " << hsum(results) << ", time: " << approx_time << "\n";

	result = 0.0f;
	next = 0.62456735f;

	// Benchmark std::sin
	timer.start();
	for (int i = 0; i < N; ++i) {
		float x = i * scaleFactor;
		result += std::sin(x);
	}
	timer.stop();
	double std_time = timer.elapsedMilliseconds();

	std::cout << "std::sin next: " << result << ", time: " << std_time << "\n";

	return 0;
}
*/




int main() {

	Graph graph{};
	Line l1(olc::RED);
	Line l2(olc::BLUE);

	for (float x = -10.0f; x <= 10.0f; x += 1e-3) {
		l1.addPoint(Point(x, approx_sin(x)));
		l2.addPoint(Point(x, std::sin(x)));
	}

	graph.addLine(l1);
	graph.addLine(l2);

	graph.waitFinish();

	return 0;
}
