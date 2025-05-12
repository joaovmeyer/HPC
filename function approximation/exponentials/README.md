# Approximating Exponentials

## Introduction

There's no doubt that exponential functions are important and commonly used in computer programs. Personally, I've used them in neural network activation functions, where speed matters and accuracy isn't always critical. That's exactly the kind of situation where approximations shine. 

The problem with standard implementations of exponential functions (and most other functions, really) is that they are often required to approximate them up to machine precision, which means you can't get a more accurate result without switching number types. That's great if precision is what you need, and it doesn't hurt otherwise, but performance can suffer from expensive calculations. 

When precision is still important, approximations can also be used: being able to make a vectorized implementation (with SIMD) means that we can calculate multiple values at once and still get speedups, even without losing too much accuracy.


## Binary Numbers Representation

Here's a quick reminder of how numbers are represented in binary:

 - Integers: an integer $i$ represented in binary with bits $b_{0}$, ..., $b_{n}$ has the value $i = \sum_{j=0}^{n} 2^{j} b_{j}$. For signed integers, the most significant bit ($b^{n}$) is subtracted from the rest, so it's value becomes $\sum_{j=0}^{n - 1} 2^{j} b_{j} - 2^{n} b_{n}$
 - Fixed point decimals: for a fractional number $0 \le d < 1$, we can represent it in a similar way, but with negative exponents: $d = \sum_{j=0}^{n} 2^{-j - 1} b_{j}$
 - Floating point numbers: a floating-point number $x$ is represented as $(-1)^{s} \cdot (1 + m) \cdot 2^{e - b}$, where $s$ is the sign bit (0 for positive, 1 for negative), $m$ is the mantissa, represented as a fixed point decimal, $e$ is the exponent, represented as an integer, and $b$ is called exponent bias, a constant term.

In IEEE 754 single-precision floats, the mantissa has 23 bits, the exponent has 8 bits, and the exponent bias is 127.


## Methods

We only need to approximate a single exponential function in order to approximate all others (since $a^{x} = b^{x \cdot \log_b(a)}$). With that, we can approximate $2^{x}$ and get any other exponential with no additional effort. The first idea to approximate $2^{x}$ is taking advantage of how numbers are stored: 

If $i$ is a non-negative integer, we know that $2^{i}$ is also a non-negative integer. That means we can represent it in its binary form as $\sum_{j=0}^{n} 2^{j} b_{j}$. It's easy to see that in that case, if we construct a binary number where only the $i$-th bit is set to 1 and all other bits set to 0, the expression simplifies to $2^{i}$, the exact value we want. This means we don't actually have to calculate $2^{i}$, we only need to set the correct bit to 1 and the result will come automatically.

```cpp
float approx_exp2_v1(float x) {
	// cast x to a positive integer
	uint32_t i = static_cast<uint32_t>(x);
	
	// this places 1 on bit i, and leaves the rest as 0
	uint32_t exp2 = 1 << i;
	
	// cast back to float
	return static_cast<float>(exp2);
}
```

And here is how it looks:

![1231b032-4f44-48a5-8727-437463bb23a1](https://github.com/user-attachments/assets/5fba11d1-2499-4886-962e-041a61422250)


This approximation is unfortunately really bad: first of all, it only works with non-negative inputs and is accurate only for integer values. Due to the rapid growth of the exponential function, it'll rapidly lose precision when the decimal part of x gets closer to 1. The relative error of this approximation can be measured like

$$ E_{r} = \frac{|V_{e} - V_{c}|}{|V_{e}|} $$

where $E_r$ is the relative error, $V_e$ is the expected value, and $V_c$ is the calculated value. It will be close to 50% at its worst:

Suppose that $x = i + d$, where $i$ is an integer and $d$ is close to (but smaller than) $1$:

$$ V_{e} \approx 2^{i + 1} = 2 \cdot 2^{i} $$

$$ V_{c} = 2^{i} $$

$$ E_{r} \approx \frac{|2 \cdot 2^{i} - 2^{i}|}{|2 \cdot 2^{i}|} = \frac{|2^{i}|}{2 \cdot |2^{i}|} = \frac{1}{2} $$


meaning the calculated value can be nearly twice as small as it should be. 

One way to improve accuracy would be to linearly interpolate between integer values, so calculate $2^{i}$ and $2^{i + 1}$ and interpolate: 

```cpp
float approx_exp2_v2(float x) {
	// cast x to an integer
	uint32_t i = static_cast<uint32_t>(x);
	float d = x - static_cast<float>(i);
	
	// calculate pow(2, i) and pow(2, i + 1)
	float exp2_1 = static_cast<float>(1 << i);
	float exp2_2 = static_cast<float>(1 << (i + 1));
	
	// linearly interpolate between the two results (this is known as 'lerp' function)
	return exp2_1 * (1.0f - d) + exp2_2 * d;
}
```

![7810f1d1-181d-405f-b3bb-7d630614dc56](https://github.com/user-attachments/assets/32671685-39e5-41bc-98a6-ae9899a18adc)


This approximation is actually much better (but it still can't handle negative values), having a maximum relative error of $\frac{2}{e \cdot \ln(2)} - 1$ (about 6.15%). This would be fine for some applications, and it's already quite fast, but it can be faster: first, note that the interpolation is mathematically equivalent to computing $2^{i} \cdot (1 + d)$:

$$ 2^{i} \cdot (1 - d) + 2^{i + 1} \cdot d = 2^{i} \cdot (1 - d) + 2^{i} \cdot 2d $$

$$ = 2^{i} \cdot ((1 - d) + 2d) = 2^{i} \cdot (1 + d) $$

This would already save us some operations, but we can do better: we know that float numbers are stored in the format $(-1)^{s} \cdot (1 + m) \cdot 2^{e - 127}$. The idea is that if we could make an integer representation of $x$, add 127 to it (to cancel the exponential bias), place it in the exponent bits of a float, and leave the temaining bits as 0, the result would have value $(-1)^{0} \cdot (1 + 0) \cdot 2^{i} = 2^{i}$:

```cpp
float approx_exp2_v3(float x) {

	int i = static_cast<int>(x);
	
	// there are 23 bits in the mantissa, so we shift
	// everything by 23 to place i + 127 in the exponent bits
	i = (i + 127) << 23;

	// reinterpret bits of i as a float. This means the exponent
	// will be i + 127, and the result will have value pow(2, i)
	float exp2 = std::bit_cast<float>(i);
	
	return exp2;
}
```

This is actually equivalent to our first version, so not much has improved (though now we correctly handle negative values, as long as i + 127 is non-negative). But here's the key: if we perform $(x + 127) \cdot (1 \ll 23)$ and only after cast it to an integer, the first 23 bits of the result will not actually be 0, they'll represent the decimal part of x, that turned to the integer part after multiplying by $2^{23}$. When multiplied by $2^{23}$, the mantissa becomes a standard 23-bit unsigned integer, thus, it will occupy the first 23 bits after casting to an integer: 

$$ m = \sum_{j=0}^{22} 2^{-j - 1} \cdot b_{j} = \frac{\sum_{j=0}^{22} 2^{22 - j} \cdot b_{j}}{2^{23}} $$

$$ \implies m * 2^{23} = \sum_{j=0}^{22} 2^{22 - j} \cdot b_{j} = \sum_{j=0}^{22} 2^{j} \cdot b_{22 - j} $$

**OBS**: note that the indices of the bits get swapped. This is totally expected, since for integers, bit 0 is the rightmost bit while for fixed point decimals, bit 0 is the leftmost bit, in the computer representation.

This means that, when reinterpreting the result back to a float, it's mantissa will not be 0, but actually an approximation (up to numerical precision) of x's decimal part, and our result will actually be close to $(-1)^{0} \cdot (1 + d) * 2^{i} = 2^{i} \cdot (1 + d)$. As we've seen, this produces the same result as the linear interpolation version:

```cpp
float approx_exp2_v4(float x) {

	// multiplying by 1 << 23 is equivalent to shifting the
	// integer by 23, but the mantissa will not be wasted
	x = (x + 127) * (1 << 23);
	uint32_t i = static_cast<uint32_t>(x);

	// bit cast will make a float with the same bits as i
	float exp2 = std::bit_cast<float>(i);
	
	return exp2;
}
```

This technique was proposed by Nicol N. Schraudolph in [this paper](https://nic.schraudolph.org/pubs/Schraudolph99.pdf) and is probably the fastest decent approximation of exponential functions. The paper also goes in depth on how to use an correction term to reduce the relative error of this approximation from about 6% to about 3%, and it's definitely worth a read. This approximation is not only super fast, but it can also approximate $2^{x}$ for negative values of x, something the first two versions couldn't:

![dee856b3-45ee-4749-8489-55b0a53e5236](https://github.com/user-attachments/assets/466e0c60-c62e-478b-bca8-68958210554a)

To improve the accuracy further, we can considere the decomposition:

$$ 2^{x} = 2^{i} \cdot 2^{d} $$

We can approximate $2^{d}$ using polynomial fitting, and then multiply it by $2^{i}$, that we already know how to compute. This could look something like this:

```cpp
float approx_exp2_v5_8(float x) {

	// we need floor so it will round the same way for positive and negative inputs
	float fi = std::floor(x);

	int i = static_cast<int>(fi);
	i = (i + 127) << 23;

	float exp2 = std::bit_cast<float>(i);

	// approximation of pow(2, x) in [0, 1) with a polynomial
	exp2 *= approx_exp2__0_1(x - fi);

	return exp2;
}
```

There are various methods for approximating $2^{d}$ with polynomials. I've personally used the well known least squares method for polynomials of degree 2 up to degree 5. At degree 5, there's barely any difference in std::exp2 and the approximation (you can see the polynomial coefficients in the code files). Here are benchmarks performed both for latency (scalar code with dependencies across iterations) and throughput (uses AVX)

![95d610eb-fde8-4215-98d1-97fe3d1bc820](https://github.com/user-attachments/assets/083f0d0f-6c38-4c97-b12a-8fd90fc36931)

The code for generating benchmarks was compiled with GCC version 13.2.0 and flags `std=c++20`, `-march=native` (sandybridge), `O2` and is available above. For calculating the speedups, I measured the time each version took to approximate exp2 for 100000000 values. Unfortunately, my CPU does not have AVX2, so I couldn't really benchmark version 3 with SIMD. Also, versions 5 through 8 could be made a little more efficient with AVX2.

Notice how the trade-off between accuracy X speed is clearly represented in the graphs: for the scalar version, measuring latency, we can have results up to 2x smaller than they should be, but 15x faster to calculate, while the most precise version, while still inferior to the standard, is "only" about 3x faster to calculate. For bulk calculations, the performance gains are even more pronounced: we can have from 60x faster imprecise values to 40x faster precise values. Even very decent results can get huge speedups with SIMD, that's great!

Still, using these approximations should be carefully considered: they can't handle numbers like NaN or infs. They also don't have a range check: if the input is not within allowed range, results would not approximate $2^{x}$ as we expect it to.

## Appendix: Derivation of Maximum Error for Linear Interpolation

In case you are interested in how I got to the formula $\frac{2}{e \cdot \ln(2)} - 1$ for maximum relative error of the linear interpolation versions, you can check the derivation for it:

Let $x = i + d$, where i is an integer and $0 \le d < 1$

$$ E_{r} = \frac{|2^{x} - 2^{i} \cdot (1 + d)|}{|2^{x}|} = \frac{|2^{i} \cdot 2^{d} - 2^{i} \cdot (1 + d)|}{|2^{i} \cdot 2^{d}|} $$

$$ = \frac{|2^{i}| \cdot |2^{d} - (1 + d)|}{|2^{i}| \cdot |2^{d}|} = \frac{|2^{d} - (1 + d)|}{|2^{d}|} = \frac{(1 + d) - 2^{d}}{2^{d}} = \frac{1 + d}{2^{d}} - 1 $$

To find the maximum relative error, we need to maximize $\frac{1 + d}{2^{d}}$ for $0 \le d < 0$. Taking the derivative and setting it to zero, we get:

$$ \frac{d}{dd} \frac{1 + d}{2^{d}} = \frac{1}{2^{d}} \cdot (1 - \ln(2) - x \cdot \ln(2)) $$

$$ \frac{d}{dd} \frac{1 + d}{2^{d}} = 0 \Leftrightarrow d = \frac{1}{\ln(2)} - 1 $$

Substituting $d = \frac{1}{\ln(2)} - 1$ into our relative error formula, we get

$$ E_{r} \le \frac{1 + \frac{1}{\ln(2)} - 1}{2^{\frac{1}{\ln(2)} - 1}} - 1 = \frac{1}{\ln(2)} \cdot \frac{1}{2^{\frac{1}{\ln(2)} - 1}} - 1 $$

$$ = \frac{1}{\ln(2)} \cdot \frac{1}{2^{\frac{1}{\ln(2)}} \cdot 2^{-1}} - 1 = \frac{2}{\ln(2) \cdot e} - 1 $$
