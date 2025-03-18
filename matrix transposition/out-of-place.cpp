#include <iostream>
#include <x86intrin.h>

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




void fill(float* a, int N) {
	for (int i = 0; i < N; ++i) a[i] = (float) rng::fromNormalDistribution(-1.0, 1.0);
}

bool cmp(const float* a, const float* b, int N) {
	for (int i = 0; i < N; ++i) if (a[i] != b[i]) return false;
	return true;
}



// 1- NAIVE TRANSPOSITION

void transpose_naive(const float* a, float* b, int rows, int cols, int lda, int ldb) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			b[j * ldb + i] = a[i * lda + j];
		}
	}
}



// 2 - CACHE-AWARE TRANSPOSITION (LOOP BLOCKING)

// the idea is to divide the matrix into multiple parts of size {block_size}x{block_size}
// such that the L1 cache has space for 2 * block_size * block_size elements. This way,
// we can transpose each block individually and reduce cache misses. One limitation is
// we ideally need to know the L1 cache size, to make the block_size as big as possible

// uses SIMD to transpose a 4x4 matrix
void transpose_4x4(const float* a, float* b, int lda, int ldb) {
	__m128 row1 = _mm_loadu_ps(a + 0 * lda);
	__m128 row2 = _mm_loadu_ps(a + 1 * lda);
	__m128 row3 = _mm_loadu_ps(a + 2 * lda);
	__m128 row4 = _mm_loadu_ps(a + 3 * lda);

	_MM_TRANSPOSE4_PS(row1, row2, row3, row4);

	_mm_storeu_ps(b + 0 * ldb, row1);
	_mm_storeu_ps(b + 1 * ldb, row2);
	_mm_storeu_ps(b + 2 * ldb, row3);
	_mm_storeu_ps(b + 3 * ldb, row4);
}

// 4x4 blocks for better SIMD utilization (use on smaller matrices)
void transpose_blocked4x4(const float* a, float* b, int rows, int cols, int lda, int ldb) {

	int i;
	for (i = 0; i + 3 < rows; i += 4) {
		int j;
		for (j = 0; j + 3 < cols; j += 4) {
			transpose_4x4(a + (i * lda + j), b + (j * ldb + i), lda, ldb);
		}

		if (j < cols) { // handle remaining columns
			transpose_naive(a + (i * lda + j), b + (j * ldb + i), 4, cols - j, lda, ldb);
		}
	}

	if (i < rows) { // handle remaining rows
		int j;
		for (j = 0; j + 3 < cols; j += 4) {
			transpose_naive(a + (i * lda + j), b + (j * ldb + i), rows - i, 4, lda, ldb);
		}

		if (j < cols) {
			transpose_naive(a + (i * lda + j), b + (j * ldb + i), rows - i, cols - j, lda, ldb);
		}
	}
}

// bigger block for better cache utilization (use on bigger matrices)
void transpose_blocked(const float* a, float* b, int rows, int cols, int lda, int ldb) {

	static constexpr int block_size = 128; // should be adjusted for the hardware it's running on

	for (int i = 0; i < rows; i += block_size) {

		// how many rows left in this block
		int r = std::min(block_size, rows - i);

		for (int j = 0; j < cols; j += block_size) {

			// how many columns left in this block
			int c = std::min(block_size, cols - j);

			transpose_blocked4x4(a + (i * lda + j), b + (j * ldb + i), r, c, lda, ldb);
		}
	}
}


// 3 - CACHE-OBLIVIOUS TRANSPOSITION (RECURSIVE)

// the oblivious algorithm can work well even without knowing the cache size
// it recursively splits the matrix in 4 parts and uses the following formula to get the transpose
// A = | A_11 A_12 | => A^T = | A_11^T A_21^T |
//     | A_21 A_22 |          | A_12^T A_22^T |

// the idea is that at some point the matrix will fit in the cache, and even if we divide it a few 
// more times than needed, eventually the transposition will be done in a cache-friendly way

// this could be problematic if abs(rows - cols) is big
void transpose_oblivious(const float* a, float* b, int rows, int cols, int lda, int ldb) {

	// stop recursion when matrices definitely fit in the cache (8KB should fit two 32x32 matrices, so 16x16 is safe)
	if (rows <= 16 && cols <= 16) {
		transpose_naive(a, b, rows, cols, lda, ldb);
		return;
	}

	// make halfRow and halfCol multiple of 4 (seems to help with SIMD usage a little)
	int halfRow = (rows / 2) & (~0b11);
	int halfCol = (cols / 2) & (~0b11);

	transpose_oblivious(a, b, halfRow, halfCol, lda, ldb);
	transpose_oblivious(a + halfCol, b + (halfCol * ldb), halfRow, cols - halfCol, lda, ldb);
	transpose_oblivious(a + (halfRow * lda), b + halfRow, rows - halfRow, halfCol, lda, ldb);
	transpose_oblivious(a + (halfRow * lda + halfCol), b + (halfCol * ldb + halfRow), rows - halfRow, cols - halfCol, lda, ldb);
}

// better if number of rows and colums are much different
void transpose_oblivious2(const float* a, float* b, int rows, int cols, int lda, int ldb) {

	// stop recursion when matrices definitely fit in the cache (8KB should fit two 32x32 matrices, so 16x16 is safe)
	if (rows <= 16 && cols <= 16) {
		// try to use some SIMD
		transpose_blocked4x4(a, b, rows, cols, lda, ldb);
	} else if (rows <= 8) { // split only on the columns

		int halfCol = (cols / 2) & (~0b11);

		transpose_oblivious2(a, b, rows, halfCol, lda, ldb);
		transpose_oblivious2(a + halfCol, b + (halfCol * ldb), rows, cols - halfCol, lda, ldb);
	} else if (cols <= 8) { // split only on the rows

		int halfRow = (rows / 2) & (~0b11);

		transpose_oblivious2(a, b, halfRow, cols, lda, ldb);
		transpose_oblivious2(a + (halfRow * lda), b + halfRow, rows - halfRow, cols, lda, ldb);
	} else {

		int halfRow = (rows / 2) & (~0b11);
		int halfCol = (cols / 2) & (~0b11);

		transpose_oblivious2(a, b, halfRow, halfCol, lda, ldb);
		transpose_oblivious2(a + halfCol, b + (halfCol * ldb), halfRow, cols - halfCol, lda, ldb);
		transpose_oblivious2(a + (halfRow * lda), b + halfRow, rows - halfRow, halfCol, lda, ldb);
		transpose_oblivious2(a + (halfRow * lda + halfCol), b + (halfCol * ldb + halfRow), rows - halfRow, cols - halfCol, lda, ldb);
	}
}







int main() {

	int rows, cols, iter;
	cout << "Rows: "; cin >> rows;
	cout << "Columns: "; cin >> cols;
	cout << "Iter: "; cin >> iter;

	int N = rows * cols;

	float* a = new float[N];
	float* b = new float[N];
	float* c = new float[N];
	float* d = new float[N];

	fill(a, N);

	Timer timer{};

	timer.start();
	for (volatile int i = 0; i < iter; ++i) transpose_blocked(a, b, rows, cols, cols, rows);
	timer.stop();
	cout << "Blocked time (ms): " << timer.elapsedTime() << "\n";

	timer.start();
	for (volatile int i = 0; i < iter; ++i) transpose_oblivious2(a, c, rows, cols, cols, rows);
	timer.stop();
	cout << "Oblivious time (ms): " << timer.elapsedTime() << "\n";

	timer.start();
	for (volatile int i = 0; i < iter; ++i) transpose_naive(a, d, rows, cols, cols, rows);
	timer.stop();
	cout << "Naive time (ms): " << timer.elapsedTime() << "\n";

	if (cmp(b, d, N)) cout << "Certo!\n";
	else cout << "Errado :(\n";

	if (cmp(c, d, N)) cout << "Certo!\n";
	else cout << "Errado :(\n";

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;

	return 0;
}