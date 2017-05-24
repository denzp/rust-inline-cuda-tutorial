# Chapter 0
> Preparation for the real things.

## Overview
We are going to define a problem, which we will later solve with CUDA. Also reference CPU implementation must be done here, to ensure correctness and benchmarking of the GPGPU implementation.

## The problem
GPGPU applies pretty well to highly parallelizable tasks. We are going to implement one of image processing algorithms.
Because, usually as many threads available, as better the image processing performance can be achieved, since in many cases every pixel could be processed separately from others.

The chosen one is a **Bilateral filtering**, which is a noise reduction algorithm. We won't try to apply algorithm optimisations because we are focusing on CUDA programming with Rust here ;)

## The algorithm
Let's stick to the naive algorithm from [Wikipedia]. We are interested in final formula interpretation with Gaussian function as kernel, which is pretty straightforward :)

![Filter formula](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20I_d%28i%2C%20j%29%20%3D%20%5Cfrac%20%7B%20%5Csum_%7Bk%2C%20l%7D%20I_s%28k%2C%20l%29%20*%20w%28i%2C%20j%2C%20k%2C%20l%29%20%7D%20%7B%20%5Csum_%7Bk%2C%20l%7D%20w%28i%2C%20j%2C%20k%2C%20l%29%20%7D)

![Filter weight](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20w%28i%2C%20j%2C%20k%2C%20l%29%20%3D%20e%5E%7B%20%28-%5Cfrac%20%7B%28i-k%29%5E2%20&plus;%20%28j-l%29%5E2%7D%20%7B2%20%5Csigma_d%5E2%7D%20-%20%5Cfrac%20%7B%5C%7CI_s%28i%2C%20j%29%20-%20I_s%28k%2C%20l%29%5C%7C%5E2%7D%7B2%20%5Csigma_r%5E2%7D%29%7D)

![Formula constraints](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Cbegin%7Barray%7D%7Blcl%7D%20k%20%26%20%5Cin%20%26%20%5Cleft%5B%20i%20-%20%5COmega%2C%20i%20&plus;%20%5COmega%20%5Cright%5D%20%5C%5C%20l%20%26%20%5Cin%20%26%20%5Cleft%5B%20j%20-%20%5COmega%2C%20j%20&plus;%20%5COmega%20%5Cright%5D%20%5C%5C%20%5COmega%20%26%20%5Cin%20%26%20%5Cmathbb%7BN%7D%20%5Cend%7Barray%7D)

[Wikipedia]: https://en.wikipedia.org/wiki/Bilateral_filter

## Results
We made a playground for our later experiments with CUDA. Currently, our reference implementation of the algorithm has been implemented in a sequential and parallel way (almost without differences).

Results are quite predictable: four physical core **Intel i5-4690K** gives us robust **4x** speedup in parallel implementation with 4 threads, compared to sequential one. So, I'm looking forward to seeing how faster the CUDA implementation will be :)

![Performance plot](../plots/chapter-0-performance.png)

![Speedup plot](../plots/chapter-0-speedup.png)