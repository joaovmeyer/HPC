#include <cmath>
#include <vector>
#include <benchmark/benchmark.h>
#include "classical.hpp"





static void naive(benchmark::State& state) {

    double x_min = -3.14159265359;
    double x_max = 3.14159265359;
    int N_points = 1000;

    std::vector<double> x; x.reserve(N_points);
    std::vector<double> y; y.reserve(N_points);
    for (int i = 0; i < N_points; ++i) {
        double x_val = x_min + (x_max - x_min) * static_cast<double>(i) / static_cast<double>(N_points - 1);
        x.push_back(x_val);
        y.push_back(std::sin(x_val));
    }

    double eval_point = 0.785398163397; // pi / 4

    for (auto _ : state) {
        double result = naive_Lagrange(x, y, eval_point);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(naive);




static void SIMD(benchmark::State& state) {

    double x_min = -3.14159265359;
    double x_max = 3.14159265359;
    int N_points = 1000;

    std::vector<double> x; x.reserve(N_points);
    std::vector<double> y; y.reserve(N_points);
    for (int i = 0; i < N_points; ++i) {
        double x_val = x_min + (x_max - x_min) * static_cast<double>(i) / static_cast<double>(N_points - 1);
        x.push_back(x_val);
        y.push_back(std::sin(x_val));
    }

    double eval_point = 0.785398163397; // pi / 4

    for (auto _ : state) {
        double result = SIMD_Lagrange(x, y, eval_point);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(SIMD);




static void kernel(benchmark::State& state) {

    double x_min = -3.14159265359;
    double x_max = 3.14159265359;
    int N_points = 1000;

    std::vector<double> x; x.reserve(N_points);
    std::vector<double> y; y.reserve(N_points);
    for (int i = 0; i < N_points; ++i) {
        double x_val = x_min + (x_max - x_min) * static_cast<double>(i) / static_cast<double>(N_points - 1);
        x.push_back(x_val);
        y.push_back(std::sin(x_val));
    }

    double eval_point = 0.785398163397; // pi / 4

    for (auto _ : state) {
        double result = kernel_Lagrange(x, y, eval_point);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(kernel);

BENCHMARK_MAIN();