#include <iostream>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <x86intrin.h>
#include <iomanip>

#include "../../graph.h"




template <typename FUNC>
void plotFunc(const FUNC& f, Graph* graph, float minX, float maxX, float step = 0.1f, olc::Pixel color = olc::BLACK) {
	Line l(color);

	for (float x = minX; x <= maxX; x += step) {
		l.addPoint(Point(x, f(x)));
	}

	graph->addLine(l);
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
	x = (x + 127.0f) * (1 << 23);
	uint32_t i = static_cast<uint32_t>(x);

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

	// approximation of 2^x - 1 - x in [0, 1)
	exp2 *= (0.342656060127262f * x + 0.649426903806752f) * x + 1.003762902276502f;

	return exp2;
}

float approx_exp2_v6(float x) {

	float fi = std::floor(x);

	int i = static_cast<int>(fi);
	x -= fi;
	i = (i + 127) << 23;

	float exp2 = std::bit_cast<float>(i);

	// approximation of 2^x - 1 - x in [0, 1)
	exp2 *= ((0.079019886937600f * x + 0.224126229720834f) * x + 0.696838835969343f) * x + 0.999811907929616f;

	return exp2;
}

float approx_exp2_v7(float x) {

	float fi = std::floor(x);

	int i = static_cast<int>(fi);
	x -= fi;
	i = (i + 127) << 23;

	float exp2 = std::bit_cast<float>(i);

	// approximation of 2^x - 1 - x in [0, 1)
	exp2 *= (((0.013676523800069f * x + 0.051666839337489f) * x + 0.241710331749456f) * x + 0.692931257740765f) * x + 1.000007286841045f;

	return exp2;
}

float approx_exp2_v8(float x) {

	float fi = std::floor(x);

	int i = static_cast<int>(fi);
	x -= fi;
	i = (i + 127) << 23;

	float exp2 = std::bit_cast<float>(i);

	// approximation of 2^x - 1 - x in [0, 1)
	exp2 *= ((((0.001894376824293f * x + 0.008940581738581f) * x + 0.055876565615224f) * x + 0.240131684394877f) * x + 0.693156778791506f) * x + 0.999999769472682f;

	return exp2;
}


float std_exp2(float x) {
	return std::exp2(x);
}



int main() {

	int n;
	cout << "Which function to plot? (1-8)";
	cin >> n;

	Graph graph{};
	graph.setXAxis(-5.5, 5.5);
	graph.setYAxis(-0.5, std::exp2(5.0) + 0.5);

	plotFunc(std_exp2, &graph, -5.0f, 5.0f, 0.01f, olc::BLUE);

	if (n == 1) plotFunc(approx_exp2_v1, &graph, -5.0f, 5.0f, 0.01f, olc::RED);
	else if (n == 2) plotFunc(approx_exp2_v2, &graph, -5.0f, 5.0f, 0.01f, olc::RED);
	else if (n == 3) plotFunc(approx_exp2_v3, &graph, -5.0f, 5.0f, 0.01f, olc::RED);
	else if (n == 4) plotFunc(approx_exp2_v4, &graph, -5.0f, 5.0f, 0.01f, olc::RED);
	else if (n == 5) plotFunc(approx_exp2_v5, &graph, -5.0f, 5.0f, 0.01f, olc::RED);
	else if (n == 6) plotFunc(approx_exp2_v6, &graph, -5.0f, 5.0f, 0.01f, olc::RED);
	else if (n == 7) plotFunc(approx_exp2_v7, &graph, -5.0f, 5.0f, 0.01f, olc::RED);
	else if (n == 8) plotFunc(approx_exp2_v8, &graph, -5.0f, 5.0f, 0.01f, olc::RED);

	graph.waitFinish();

	return 0;
}