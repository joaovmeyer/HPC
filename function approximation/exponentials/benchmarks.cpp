#include <iostream>
#include <cmath>
#include <cstdint>
#include <x86intrin.h>
#include <iomanip>
#include <chrono>







// this measures latency, not throughput, by making each call depend on the last
template <typename FUNC>
double timeFunc(const FUNC& f, int N) {

    float next = -30.0f + static_cast<float>(std::rand()) / (RAND_MAX + 1.0f) * 30.0f;

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < N; ++i) {
		// multiply by something <= 0.5 so it doesn't diverge
		next = f(next) * 0.47931f;
	}

	auto end = std::chrono::high_resolution_clock::now();

	return std::chrono::duration<double, std::milli>(end - start).count();
}





float approx_exp2_v1(float x) {
	// cast x to an positive integer
	uint32_t i = static_cast<uint32_t>(x);
	
	// calculate pow(2, i)
	uint32_t exp2 = 1 << i;
	
	// cast back to float
	return static_cast<float>(exp2);
}

float approx_exp2_v2(float x) {
	// cast x to an integer
	uint32_t i = static_cast<uint32_t>(x);
	float d = x - static_cast<float>(i);
	
	// calculate pow(2, i) and pow(2, i + 1)
	float exp2_1 = static_cast<float>(1 << i);
	float exp2_2 = static_cast<float>(1 << (i + 1));
	
	// linearly interpolate both results
	return exp2_1 * (1.0f - d) + exp2_2 * d;
}

float approx_exp2_v3(float x) {

	int i = static_cast<int>(x);
	
	// there are 23 bits in the mantissa, so we shift
	// everything by 23 to place i + 127 in the exponent bits
	i = (i + 127) << 23;

	// reinterpret bits of i as a float. This will mean the 
	// exponent will be i + 127, and x will have value pow(2, i)
	float exp2 = std::bit_cast<float>(i);
	
	return exp2;
}

float approx_exp2_v4(float x) {

	// multiplying by 1 << 23 is equivalent to shifting the
	// integer by 23, but the mantissa will not be wasted
	x *= 1 << 23;
	uint32_t i = static_cast<uint32_t>(x);

	// integer addition is a little faster, so only add here (actually about 14% faster)
	i += 127 << 23;

	// bit cast will make a float with the same bits as i
	float exp2 = std::bit_cast<float>(i);
	
	return exp2;
}

float approx_exp2_v5(float x) {

	float fi = std::floor(x);

	int i = static_cast<int>(fi);
	x -= fi;
	i = (i + 127) << 23;

	float exp2 = std::bit_cast<float>(i);

	// approximation of 2^x - 1 - x in [0, 1) with polynomial of degree 2
	exp2 *= (0.342656060127262f * x + 0.649426903806752f) * x + 1.003762902276502f;

	return exp2;
}

float approx_exp2_v6(float x) {

	float fi = std::floor(x);

	int i = static_cast<int>(fi);
	x -= fi;
	i = (i + 127) << 23;

	float exp2 = std::bit_cast<float>(i);

	// approximation of 2^x - 1 - x in [0, 1) with polynomial of degree 3
	exp2 *= ((0.079019886937600f * x + 0.224126229720834f) * x + 0.696838835969343f) * x + 0.999811907929616f;

	return exp2;
}

float approx_exp2_v7(float x) {

	int i = static_cast<int>(x);
	x -= i;
	i = (i + 127) << 23;

	float exp2 = std::bit_cast<float>(i);

	// approximation of 2^x - 1 - x in [0, 1) with polynomial of degree 4
	exp2 *= (((0.013676523800069f * x + 0.051666839337489f) * x + 0.241710331749456f) * x + 0.692931257740765f) * x + 1.000007286841045f;

	return exp2;
}

float approx_exp2_v8(float x) {

	float fi = std::floor(x);

	int i = static_cast<int>(fi);
	x -= fi;
	i = (i + 127) << 23;

	float exp2 = std::bit_cast<float>(i);

	// approximation of 2^x - 1 - x in [0, 1) with polynomial of degree 5
	exp2 *= ((((0.001894376824293f * x + 0.008940581738581f) * x + 0.055876565615224f) * x + 0.240131684394877f) * x + 0.693156778791506f) * x + 0.999999769472682f;

	return exp2;
}


float std_exp2(float x) {
	return std::exp2(x);
}



int main() {

	std::srand(std::time(0));

	int N;
	std::cout << "How many iterations? ";
	std::cin >> N;

	std::cout << "Time std (ms): " << timeFunc(std_exp2, N) << "\n";
	std::cout << "Time approximation 1 (ms): " << timeFunc(approx_exp2_v1, N) << "\n";
	std::cout << "Time approximation 2 (ms): " << timeFunc(approx_exp2_v2, N) << "\n";
	std::cout << "Time approximation 3 (ms): " << timeFunc(approx_exp2_v3, N) << "\n";
	std::cout << "Time approximation 4 (ms): " << timeFunc(approx_exp2_v4, N) << "\n";
	std::cout << "Time approximation 5 (ms): " << timeFunc(approx_exp2_v5, N) << "\n";
	std::cout << "Time approximation 6 (ms): " << timeFunc(approx_exp2_v6, N) << "\n";
	std::cout << "Time approximation 7 (ms): " << timeFunc(approx_exp2_v7, N) << "\n";
	std::cout << "Time approximation 8 (ms): " << timeFunc(approx_exp2_v8, N) << "\n";

	return 0;
}