#include <iostream>
#include <cmath>
#include <cstdint>
#include <x86intrin.h>
#include <iomanip>
#include <chrono>





// this measures latency, not throughput, by making each call depend on the last
template <typename FUNC>
double timeFunc(const FUNC& f, float* data, int N) {

	for (int i = 0; i < N; ++i) data[i] = -30.0f + static_cast<float>(std::rand()) / (RAND_MAX + 1) * 60.0f;

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i + 7 < N; i += 8) {
		f(data + i);
	}

	auto end = std::chrono::high_resolution_clock::now();

	return std::chrono::duration<double, std::milli>(end - start).count();
}


/*
void approx_exp2_v3(float* data) {

	__m256 x = _mm256_loadu_ps(data);

	__m256i i = _mm256_cvtps_epi32(x);
	i = _mm256_mul_epi32(_mm256_add_epi32(i, _mm256_set1_epi32(127)), _mm256_set1_epi32(1 << 23));

	__m256 exp2 = _mm256_castsi256_ps(i);

	_mm256_storeu_ps(data, exp2);
}*/

void approx_exp2_v4(float* data) {

	__m256 x = _mm256_loadu_ps(data);

	__m256i i = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_add_ps(x, _mm256_set1_ps(127.0f)), _mm256_set1_ps(1 << 23)));

	__m256 exp2 = _mm256_castsi256_ps(i);

	_mm256_storeu_ps(data, exp2);
}




void approx_exp2_v5(float* data) {

	__m256 x = _mm256_loadu_ps(data);

	__m256 i = _mm256_floor_ps(x);
	__m256 d = _mm256_sub_ps(x, i);

	__m256 exp2 = _mm256_castsi256_ps(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_add_ps(i, _mm256_set1_ps(127.0f)), _mm256_set1_ps(1 << 23))));

	exp2 = _mm256_mul_ps(exp2, 
		_mm256_add_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(
			_mm256_set1_ps(0.342656060127262f), d), 
			_mm256_set1_ps(0.649426903806752f)), d), 
			_mm256_set1_ps(1.003762902276502f))
	);

	_mm256_storeu_ps(data, exp2);
}

void approx_exp2_v6(float* data) {

	__m256 x = _mm256_loadu_ps(data);

	__m256 i = _mm256_floor_ps(x);
	__m256 d = _mm256_sub_ps(x, i);

	__m256 exp2 = _mm256_castsi256_ps(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_add_ps(i, _mm256_set1_ps(127.0f)), _mm256_set1_ps(1 << 23))));

	exp2 = _mm256_mul_ps(exp2, 
		_mm256_add_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(
			_mm256_set1_ps(0.079019886937600f), d), 
			_mm256_set1_ps(0.224126229720834f)), d), 
			_mm256_set1_ps(0.696838835969343f)), d), 
			_mm256_set1_ps(0.999811907929616f))
	);

	_mm256_storeu_ps(data, exp2);
}

void approx_exp2_v7(float* data) {

	__m256 x = _mm256_loadu_ps(data);

	__m256 i = _mm256_floor_ps(x);
	__m256 d = _mm256_sub_ps(x, i);

	__m256 exp2 = _mm256_castsi256_ps(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_add_ps(i, _mm256_set1_ps(127.0f)), _mm256_set1_ps(1 << 23))));

	exp2 = _mm256_mul_ps(exp2, 
		_mm256_add_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(
			_mm256_set1_ps(0.013676523800069f), d), 
			_mm256_set1_ps(0.051666839337489f)), d), 
			_mm256_set1_ps(0.241710331749456f)), d), 
			_mm256_set1_ps(0.692931257740765f)), d), 
			_mm256_set1_ps(1.000007286841045f))
	);

	_mm256_storeu_ps(data, exp2);
}

void approx_exp2_v8(float* data) {

	__m256 x = _mm256_loadu_ps(data);

	__m256 i = _mm256_floor_ps(x);
	__m256 d = _mm256_sub_ps(x, i);

	__m256 exp2 = _mm256_castsi256_ps(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_add_ps(i, _mm256_set1_ps(127.0f)), _mm256_set1_ps(1 << 23))));

	exp2 = _mm256_mul_ps(exp2, 
		_mm256_add_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(
			_mm256_set1_ps(0.001894376824293f), d), 
			_mm256_set1_ps(0.008940581738581f)), d), 
			_mm256_set1_ps(0.055876565615224f)), d), 
			_mm256_set1_ps(0.240131684394877f)), d), 
			_mm256_set1_ps(0.693156778791506f)), d), 
			_mm256_set1_ps(0.999999769472682f))
	);

	_mm256_storeu_ps(data, exp2);
}


void std_exp2(float* data) {
	for (int i = 0; i < 8; ++i) {
		data[i] = std::exp2(data[i]);
	}
}





int main() {

	std::srand(std::time(0));

	int N;
	std::cout << "How many iterations? ";
	std::cin >> N;

	float* data = new float[N];

	std::cout << "Time std (ms): " << timeFunc(std_exp2, data, N) << "\n";
//	std::cout << "Time approximation 3 (ms): " << timeFunc(approx_exp2_v3, data, N) << "\n";
	std::cout << "Time approximation 4 (ms): " << timeFunc(approx_exp2_v4, data, N) << "\n";
	std::cout << "Time approximation 5 (ms): " << timeFunc(approx_exp2_v5, data, N) << "\n";
	std::cout << "Time approximation 6 (ms): " << timeFunc(approx_exp2_v6, data, N) << "\n";
	std::cout << "Time approximation 7 (ms): " << timeFunc(approx_exp2_v7, data, N) << "\n";
	std::cout << "Time approximation 8 (ms): " << timeFunc(approx_exp2_v8, data, N) << "\n";

	delete[] data;

	return 0;
}