# Lagrange Interpolation

For now, this implements the "classical" Lagrange interpolation algorithm. It includes 3 versions: the naive "textbook" version, a version using SIMD to vectorize accumulation over products and another one that uses micro-kernels, focusing on ILP and avoiding repeated calculations. I might add similar implementations for the barycentric form of Lagrange interpolation in the future.

There are probably some ways to optimize this further a little bit, but I'm more than satisfied with what I got. Blocking could have large impacts on performance if the number of points were to be much larger, but nobody does polynomial interpolation with tens of thousands of points, so I didn't even bother. It would be interesting to see the results though.

I have to thank my friend [Vinícius](https://github.com/ViniBarce) for the inspiration for this little project. The idea of optimizing Lagrange interpolation came entirely from him and my SIMD version is basically an adaptation of [his code](https://github.com/ViniBarce/ML-NumericalMethods/blob/master/Numerical%20Methods/LagrangeInterp.cpp).

Benchmarks are done with [Google Benchmark](https://github.com/google/benchmark), compiling with GCC 14.2.0 and the flags `-O3`, `-std=c++23`, `-march=native`. These are the results I got:
```txt
Run on (4 X 3093 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 256 KiB (x4)
  L3 Unified 6144 KiB (x1)
-----------------------------------------------------
Benchmark           Time             CPU   Iterations
-----------------------------------------------------
naive         7143087 ns      7114955 ns          112
SIMD           382550 ns       384976 ns         1867
kernel         102242 ns        97656 ns         6400
```

So, SIMD around ~18.6x faster than naive and kernel around ~69.8x faster than naive and ~3.7x faster than SIMD

**Note:** If I compile with `-Ofast` AND modify the naive version so that it uses two inner loops instead of checking `i == j`, its performance doubles, but numerical precision takes a significant hit. Other versions stay the same in terms of performance.
