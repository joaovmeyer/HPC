#include <vector>
#include <x86intrin.h>



// just plain simple "textboox" Lagrange interpolation
double naive_Lagrange(const std::vector<double>& x, const std::vector<double>& y, double a) {

	size_t n = x.size();
	double res = 0.0;

	for (size_t i = 0; i < n; ++i) {
		double prod = y[i];

		for (size_t j = 0; j < n; ++j) {
			if (j == i) continue;

			prod *= (a - x[j]) / (x[i] - x[j]);
		}

		res += prod;
	}

	return res;
}





inline double hmul_256(__m256d v) {
	// Permuta e multiplica: [0, 1, 2, 3] -> [0*2, 1*3, X, X]
	// __m256d v_perm = _mm256_permute4x64_pd(v, 0b10110001);

	// AVX1 approach (since we don't have to permute across lanes, this is perfect!)
	__m256d v_perm = _mm256_permute_pd(v, 0b0101);
	__m256d v_mul1 = _mm256_mul_pd(v, v_perm);

	// Extrai a parte baixa e alta para __m128d e multiplica
	__m128d v_low = _mm256_castpd256_pd128(v_mul1);
	__m128d v_high = _mm256_extractf128_pd(v_mul1, 1);
	__m128d v_res = _mm_mul_pd(v_low, v_high);

	// Extrai o escalar final
	return _mm_cvtsd_f64(v_res);
}




// basic vectorized accumulation over products (changes operations orders, so not equivalent to naive...)
double SIMD_Lagrange(const std::vector<double>& x, const std::vector<double>& y, double a) {
	size_t n = x.size();

	__m256d v_a = _mm256_set1_pd(a);
	double result = 0.0;

	for (size_t j = 0; j < n; ++j) {
		
		double xj = x[j];
		__m256d v_xj = _mm256_set1_pd(xj);
		__m256d v_num_prod = _mm256_set1_pd(1.0);
		__m256d v_den_prod = _mm256_set1_pd(1.0);
		
		double scalar_num = 1.0;
		double scalar_den = 1.0;

		size_t i;

		// handle i < j
		for (i = 0; i + 3 < j; i += 4) {
			__m256d v_xi = _mm256_loadu_pd(&x[i]);
			v_den_prod = _mm256_mul_pd(v_den_prod, _mm256_sub_pd(v_xj, v_xi));
			v_num_prod = _mm256_mul_pd(v_num_prod, _mm256_sub_pd(v_a, v_xi));
		}

		for (; i < j; ++i) {
			double xi = x[i];
			scalar_den *= (xj - xi);
			scalar_num *= (a - xi);
		}

		// handle i > j
		for (i = j + 1; i + 3 < n; i += 4) {
			__m256d v_xi = _mm256_loadu_pd(&x[i]);
			v_den_prod = _mm256_mul_pd(v_den_prod, _mm256_sub_pd(v_xj, v_xi));
			v_num_prod = _mm256_mul_pd(v_num_prod, _mm256_sub_pd(v_a, v_xi));
		}
		for (; i < n; ++i) {
			double xi = x[i];
			scalar_den *= (xj - xi);
			scalar_num *= (a - xi);
		}

		scalar_num *= hmul_256(v_num_prod);
		scalar_den *= hmul_256(v_den_prod);

		result += y[j] * (scalar_num / scalar_den);
	}

	return result;
};



// adds "micro-kernels", that serves as a form of register-blocking, saves a few
// _mm256_sub_pd(v_a, v_xi) and also enables more instruction level parallelism.
double kernel_Lagrange(const std::vector<double>& x, const std::vector<double>& y, double a) {
	size_t n = x.size();

	__m256d v_a = _mm256_set1_pd(a);
	double result = 0.0;


	static constexpr size_t B = 6;

	double xj[B];
	__m256d v_xj[B];
	__m256d v_num_prod[B];
	__m256d v_den_prod[B];

	double scalar_num[B];
	double scalar_den[B];

	size_t j;
	for (j = 0; j + B - 1 < n; j += B) {

		for (int k = 0; k < B; ++k) {
			xj[k] = x[j + k];
			v_xj[k] = _mm256_set1_pd(xj[k]);
			v_num_prod[k] = _mm256_set1_pd(1.0);
			v_den_prod[k] = _mm256_set1_pd(1.0);

			scalar_num[k] = y[j + k];
			scalar_den[k] = 1.0;
		}

		// handle i < j
		size_t i;
		for (i = 0; i + 3 < j; i += 4) {
			__m256d v_xi = _mm256_loadu_pd(&x[i]);
			__m256d v_delta_a = _mm256_sub_pd(v_a, v_xi);
			for (int k = 0; k < B; ++k) {
				__m256d v_delta_x = _mm256_sub_pd(v_xj[k], v_xi);
				v_den_prod[k] = _mm256_mul_pd(v_den_prod[k], v_delta_x);
				v_num_prod[k] = _mm256_mul_pd(v_num_prod[k], v_delta_a);
			}
		}
		for (; i < j + B; ++i) {
			double xi = x[i];
			double delta_a = a - xi;
			for (int k = 0; k < B; ++k) {
				if (i == j + k) continue;
				scalar_den[k] *= (xj[k] - xi);
				scalar_num[k] *= delta_a;
			}
		}

		// handle i > j
		for (i = j + B; i + 3 < n; i += 4) {
			__m256d v_xi = _mm256_loadu_pd(&x[i]);
			__m256d v_delta_a = _mm256_sub_pd(v_a, v_xi);
			for (int k = 0; k < B; ++k) {
				__m256d v_delta_x = _mm256_sub_pd(v_xj[k], v_xi);
				v_den_prod[k] = _mm256_mul_pd(v_den_prod[k], v_delta_x);
				v_num_prod[k] = _mm256_mul_pd(v_num_prod[k], v_delta_a);
			}
		}
		for (; i < n; ++i) {
			double xi = x[i];
			double delta_a = a - xi;
			for (int k = 0; k < B; ++k) {
				scalar_den[k] *= (xj[k] - xi);
				scalar_num[k] *= delta_a;
			}
		}

		for (int k = 0; k < B; ++k) {
			scalar_num[k] *= hmul_256(v_num_prod[k]);
			scalar_den[k] *= hmul_256(v_den_prod[k]);

			result += scalar_num[k] / scalar_den[k];
		}
	}

	// this is so much code for just handling the tail, but I don't feel like thinking of a better way of doing it
	for (; j < n; ++j) {
		
		double xj = x[j];
		__m256d v_xj = _mm256_set1_pd(xj);
		__m256d v_num_prod = _mm256_set1_pd(1.0);
		__m256d v_den_prod = _mm256_set1_pd(1.0);
		
		double scalar_num = y[j];
		double scalar_den = 1.0;

		size_t i;

		// handle i < j
		for (i = 0; i + 3 < j; i += 4) {
			__m256d v_xi = _mm256_loadu_pd(&x[i]);
			v_den_prod = _mm256_mul_pd(v_den_prod, _mm256_sub_pd(v_xj, v_xi));
			v_num_prod = _mm256_mul_pd(v_num_prod, _mm256_sub_pd(v_a, v_xi));
		}

		for (; i < j; ++i) {
			double xi = x[i];
			scalar_den *= (xj - xi);
			scalar_num *= (a - xi);
		}

		// handle i > j
		for (i = j + 1; i + 3 < n; i += 4) {
			__m256d v_xi = _mm256_loadu_pd(&x[i]);
			v_den_prod = _mm256_mul_pd(v_den_prod, _mm256_sub_pd(v_xj, v_xi));
			v_num_prod = _mm256_mul_pd(v_num_prod, _mm256_sub_pd(v_a, v_xi));
		}
		for (; i < n; ++i) {
			double xi = x[i];
			scalar_den *= (xj - xi);
			scalar_num *= (a - xi);
		}

		scalar_num *= hmul_256(v_num_prod);
		scalar_den *= hmul_256(v_den_prod);

		result += scalar_num / scalar_den;
	}

	return result;
}