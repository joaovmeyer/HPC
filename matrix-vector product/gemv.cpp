#include <iostream>
#include <x86intrin.h>
#include <cstring>
#include <cmath>

#include "../../rng.h"

using namespace std;




struct Timer {

	std::chrono::high_resolution_clock::time_point start_time;
	std::chrono::high_resolution_clock::time_point end_time;


	void start() {
		start_time = std::chrono::high_resolution_clock::now();
	}

	void stop() {
		end_time = std::chrono::high_resolution_clock::now();
	}


	double elapsedTime() const {
		return std::chrono::duration<double, std::milli>(end_time - start_time).count();
	}

	void displayTime() const {
		std::cout << "Elapsed time: " << elapsedTime() << " ms\n";
	}
};




float* alloc(int n) {
	return new (std::align_val_t(32)) float[n]();
}

void dealloc(float* v) {
	::operator delete[] (v, std::align_val_t(32));
}


__m256* allocAVX(int n) {
	return new (std::align_val_t(32)) __m256[n]();
}

void deallocAVX(__m256* v) {
	::operator delete[] (v, std::align_val_t(32));
}



void fill(float* a, int N) {
	for (int i = 0; i < N; ++i) a[i] = (float) rng::fromUniformDistribution(-1.0, 1.0);
}

bool cmp(const float* a, const float* b, int N) {
	static constexpr float eps = 1e-1;
	for (int i = 0; i < N; ++i) if (std::abs(a[i] - b[i]) > eps) return false;
	return true;
}



void gemv_naive(const float* a, const float* b, float* c, int rows, int cols, int lda) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			c[i] += a[i * lda + j] * b[j];
		}
	}
}



float dot_prod(const float* a, const float* b, int N) {

	__m256 vec = _mm256_setzero_ps();
	int j;
	for (j = 0; j + 7 < N; j += 8) {
		vec = _mm256_add_ps(vec, _mm256_mul_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j)));
	}

	float s = 0.0f;

	// GCC allows it
	for (int k = 0; k < 8; ++k) {
		s += vec[k];
	}

	// don't forget remaining elements
	for (; j < N; ++j) {
		s += a[j] * b[j];
	}

	return s;
}

void gemv_SIMD(const float* a, const float* b, float* c, int rows, int cols, int lda) {
	for (int i = 0; i < rows; ++i) {
		c[i] += dot_prod(a + i * lda, b, cols);
	}
}





template <int rr>
void kernel(const float* a, const float* b, float* c, int cols, int lda) {

	__m256 vecs[rr] = { _mm256_setzero_ps() };
	int j;
	for (j = 0; j + 7 < cols; j += 8) {
		__m256 vec_b = _mm256_loadu_ps(b + j);

		for (int i = 0; i < rr; ++i) {
			vecs[i] = _mm256_add_ps(vecs[i], _mm256_mul_ps(_mm256_loadu_ps(a + i * lda + j), vec_b));
		}
	}

	for (int i = 0; i < rr; ++i) {

		float s = 0.0f;

		// GCC allows it
		for (int k = 0; k < 8; ++k) {
			s += vecs[i][k];
		}

		// don't forget remaining elements
		for (int jj = j; jj < cols; ++jj) {
			s += a[i * lda + jj] * b[jj];
		}

		c[i] += s;
	}
}

void gemv_kernel(const float* a, const float* b, float* c, int rows, int cols, int lda) {

	// two seems to work best. Wierdly, it seems like register reuse is not having
	// any impact here, and making a kernel likely only helps with throughput saturation
	static constexpr int rr = 2;

	int i;
	for (i = 0; i + rr - 1 < rows; i += rr) {
		kernel<rr>(a + i * lda, b, c + i, cols, lda);
	}

	gemv_SIMD(a + i * lda, b, c + i, rows - i, cols, lda);
}


// seems to give a SMALL improvement over just the kernel past the L2 cache, but nothing crazy
// the kernel acts like a small block in some way, so this is just a little more specific
void gemv_blocked(const float* a, const float* b, float* c, int rows, int cols, int lda) {

	static constexpr int block_row = 128;
	static constexpr int block_col = 8192;

	for (int i = 0; i < rows; i += block_row) {

		// how many rows left in this block
		int row = std::min(block_row, rows - i);

		for (int j = 0; j < cols; j += block_col) {

			// how many columns left in this block
			int col = std::min(block_col, cols - j);

			gemv_kernel(a + (i * lda + j), b + j, c + i, row, col, lda);
		}
	}
}


int main() {

	int rows = 1024, cols = 500000;
	// cout << "Number of rows: "; cin >> rows;
	// cout << "Number of cols: "; cin >> cols;


	float* a = alloc(rows * cols);
	float* b = alloc(cols);
	float* c1 = alloc(rows);
	float* c2 = alloc(rows);
	float* c3 = alloc(rows);
	float* c4 = alloc(rows);

	fill(a, rows * cols);
	fill(b, cols);


	Timer timer{};


	timer.start();
	gemv_naive(a, b, c1, rows, cols, cols);
	timer.stop();
	cout << "Naive: " << timer.elapsedTime() << "\n";

	timer.start();
	gemv_SIMD(a, b, c2, rows, cols, cols);
	timer.stop();
	cout << "SIMD: " << timer.elapsedTime() << "\n";

	timer.start();
	gemv_kernel(a, b, c3, rows, cols, cols);
	timer.stop();
	cout << "Kernel: " << timer.elapsedTime() << "\n";

	timer.start();
	gemv_blocked(a, b, c4, rows, cols, cols);
	timer.stop();
	cout << "Blocked: " << timer.elapsedTime() << "\n";


	if (cmp(c1, c2, rows)) cout << "Certo!\n";
	else cout << "Errado :(\n";
	if (cmp(c1, c3, rows)) cout << "Certo!\n";
	else cout << "Errado :(\n";
	if (cmp(c1, c4, rows)) cout << "Certo!\n";
	else cout << "Errado :(\n";

	dealloc(a);
	dealloc(b);
	dealloc(c1);
	dealloc(c2);
	dealloc(c3);
	dealloc(c4);

	return 0;
}
