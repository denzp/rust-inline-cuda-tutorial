# Chapter 1
> First iteration. Separated CPU and GPU crates.

## Introduction
So, we have a reference algorithm implementation and reference filter outputs. We are now ready to dig into CUDA development.

For now, let's try the easiest approach: we are going to have independent crates with **host** and **device** code.

## Device code crate

### Prerequirements
First of all, we will need `xargo` and `ptx-linker`. Both can be installed from crates.io:
* `cargo install xargo`
* `cargo install ptx-linker`

We need `xargo` to build (because rust doesn't come with a prebuilt one) `libcore` for NVPTX target called `nvptx64-nvidia-cuda`.
Also, `ptx-linker` helps us with linking of multiple crates and can also help to avoid failed [LLVM i128 assertions - rust#38824](https://github.com/rust-lang/rust/issues/38824).

### How to compile the PTX assembly
Original instructions can be taken from [japaric/nvptx](https://github.com/japaric/nvptx), but we will use modified steps, mainly because of `ptx-linker`:

1. Create "kernels" **dylib** crate.
<br />Otherwise, rust won't call the linker.
2. Add as the crate dependency `nvptx-builtins = "0.1.0"`.
<br />We need it to access CUDA buildtins intrinsics.
3. Copy suggested [nvptx target definition](https://github.com/denzp/rust-ptx-linker/blob/master/examples/depenencies/nvptx64-nvidia-cuda.json) from [denzp/rust-ptx-linker](https://github.com/denzp/rust-ptx-linker) to the crate root.
<br />We need it because the `nvptx64-nvidia-cuda` target is not included into the Rust compiler by-default.
4. Write the code :)
5. Make rust and linker happy.
<br />Since we are building a kind of **dylib** we need to provide a dummy `panic_fmt`:
``` rust
#[lang = "panic_fmt"]
fn panic_fmt() -> ! {
    loop {}
}
```
6. Run `xargo build` with args `--release --target nvptx64-nvidia-cuda`.
7. Locate just created PTX assembly file, at `${CRATE_ROOT}/target/nvptx64-nvidia-cuda/release/${CRATE_NAME}.ptx`.

### Kernels and device functions
We need to decide which functions should be kernels, and which are just device functions.
It's important to assign kernels a special ABI with `extern "ptx-kernel"` and prevent rust from mangling their names.

Next example shows a significant difference between the two types of functions and mangling:

``` rust
// Kernel function:
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bilateral_filter(/*...*/) { }

// Kernel function:
pub unsafe extern "ptx-kernel" fn bilateral_filter2(/*...*/) { }

// Normal function:
unsafe fn w_kernel(/*...*/) -> f64 { }
```

It will be compiled into something like:

```
.visible .entry
bilateral_filter(/*...*/)

.visible .entry
_ZN16chapter_1_kernel17bilateral_filter28whateverE(/*...*/)

.func (.param .b64 retval0)
_ZN16chapter_1_kernel8w_kernel17hab5b2aabbc8bc703E(/*...*/)
```

Both first and second functions are kernels and can be executed from a GPU device, thanks to `.visible .entry`.
Hope you agree with me, that it's more convenient to call the first kernel with clear name than with second with mangled name ;)

### CUDA intrinsics
Normally we can't avoid using block and thread indices in kernels, at least because it's the core of parallelization.

In C/C++ code they can be accessed via `blockIdx.{x, y, z}`, `blockDim.{x, y, z}` and `threadIdx.{x, y, z}`.
In rust we must use [LLVM intrinsics](https://llvm.org/docs/NVPTXUsage.html#llvm-nvvm-read-ptx-sreg) which can be accessed with:

``` rust
#![feature(platform_intrinsics)]

extern "platform-intrinsic" {
    pub fn nvptx_block_idx_x() -> i32;
    pub fn nvptx_block_dim_x() -> i32;
    pub fn nvptx_thread_idx_x() -> i32;
    // ...
}
```

But we don't really need to define them in each device-side crate, we will them already defined in [japaric/nvptx-builtins](https://github.com/japaric/nvptx-builtins/blob/master/src/lib.rs).

### Absence of `std`
We have to write `#![no_std]` code because, pretty obviously, original `std` crate cannot be easily compiled for CUDA (and in most cases, you won't need it).
This means you also won't have access to any math from `std`. You can workaround math issue in couple ways.

#### LLVM intrinsics
First, you can use [LLVM and Rust intrinsics](https://github.com/rust-lang/rust/blob/afa1240e57330d85a372db4e28cd8bc8fa528ccb/src/libcore/intrinsics.rs#L1040):

``` rust
extern "rust-intrinsic" {
    fn sqrtf64(x: f64) -> f64;
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn foo(src: *const f64, dst: *mut f64) {
    *dst.offset(0) = sqrtf64(*src.offset(0))
}
```

Here, the `sqrtf64` call will be translated to a `sqrt.rn.f64` PTX instruction.

The approach looks quite okay, but it's not really so nice as it might look like.
NVPTX target doesn't support all the intrinsics that LLVM provides us. The next code can't be compiled:

``` rust
extern "rust-intrinsic" {
    fn expf64(x: f64) -> f64;
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn foo(src: *const f64, dst: *mut f64) {
    *dst.offset(0) = expf64(*src.offset(0))
}
```

with error:

```
LLVM ERROR: Cannot select: t13: i64 = ExternalSymbol'exp'
In function: foo
```

because there is no exponent PTX instruction.
[Here is a list](http://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions) of available PTX math functions available as instructions.

#### Math functions written in Rust
Alternatively, you can use any math library written in Rust and which doesn't have dependencies on `std`.

We will use the approach in this tutorial. The chosen math library is [nagisa/math.rs](https://github.com/nagisa/math.rs).

## Host code crate

Since we should already have a PTX assembly, we are free to choose (or maybe even write own) CUDA bindings.
We are interested low-level CUDA driver API.
For this tutorial, I decided to stick with [japaric/cuda](https://github.com/japaric/cuda).

Also, for convenient workflow, we could leverage [cargo build script](http://doc.crates.io/build-script.html) to build PTX assembly.
Then we won't need to run `xargo` every time we change kernel code.
The script can be seen at [`application/build_kernel.rs`](application/build_kernel.rs).

### How to load the PTX assembly
With **japaric/cuda** we need to perform next steps:

1. Call `driver::initialize()`.
2. Create `driver::Device` instance.
3. Create `driver::Context` instance.
4. Create `driver::Module` instance with PTX assembly.
5. Create `driver::Function` instance with the kernel name (your `pub extern "ptx-kernel"` function from device code crate).

We are going to provide the assembly for `driver::Module` with a help of macro: `include_str!("path/to/assembly.ptx")`. That will store PTX at compile-time and we should not care about shipping the assembly with executable.

For easier understanding of the example, we will keep CUDA stuff as statics.
It's not a production-grade approach, but enough for our small demonstration application.

``` rust
use cuda::driver::{Context, Module, Function};

pub struct StaticContext(Context);
pub struct StaticModule(Module<'static>);
pub struct StaticKernel(Function<'static, 'static>);

lazy_static! {
    pub static ref CUDA_CTX: StaticContext = StaticContext::new();
    pub static ref CUDA_MODULE: StaticModule = StaticModule::new(&CUDA_CTX);
    pub static ref CUDA_KERNEL: StaticKernel = StaticKernel::new(&CUDA_MODULE);
}
```

More details can be found at [`application/src/static_cuda.rs`](application/src/static_cuda.rs).

### Executing the kernel
To run the CUDA version of a filter from [Chapter 0](../chapter-0/README.md) we need:

1. Allocate GPU memory for the input and output images.
2. Copy input image into the respective GPU memory.
3. Decide about *grid* and *block* sizes.
4. Launch the kernel.
5. Copy output image from GPU memory into RAM.
6. Free GPU memory.

The code located at [`application/src/filter/bilateral_cuda.rs`](application/src/filter/bilateral_cuda.rs).

Here is some notes:

* At this point, we use our statics `CUDA_CTX`, `CUDA_MODULE` and `CUDA_KERNEL`.
* Since our test images have power-of-two dimensions, it's safe to just choose block size `(8, 8)` and then calculate grid as `(WIDTH / 8, HEIGHT / 8)`.
* **Don't forget!** CUDA contexts are thread-dependent. We have to call `cuCtxSetCurrent` (or `Context::set_current(&self)` in our case) if the current thread doesn't own the context.

## Results
Performance-wise the results are impressive. CUDA implementation gives us **~60-70x** speedup over sequential algorithm on **GTX 1080** GPU.

| Image resolution | Sequential processing time | Parallel processing time |
| ---------------- | -------------------------- | ------------------------ |
| 512x512          | 914.267ms                  | 15.190ms                 |
| 1024x1024        | 3815.428ms                 | 55.069ms                 |
| 2048x2048        | 14299.042ms                | 220.193ms                |
| 4096x4096        | -                          | 947.654ms                |

![Performance plot](../plots/chapter-1-performance.png)

| Image resolution | Sequential speedup | CUDA speedup |
| ---------------- | ------------------ | ------------ |
| 512x512          | 1.000              | 60.189       |
| 1024x1024        | 1.000              | 69.284       |
| 2048x2048        | 1.000              | 64.939       |

![Speedup plot](../plots/chapter-1-speedup.png)

## Next steps
In next chapters, we are going to merge both crates - device and host codebases.
Current implementation might be good enough unless we have some code to share between them.

Even in our tutorial we such case: every change on `Pixel` struct should be reflected in both crates.

So, let's go on and try to solve this inconvenience :)