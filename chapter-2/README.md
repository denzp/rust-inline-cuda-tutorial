# Chapter 2
> Merging CPU and GPU code into one crate.

## Introduction
After our first success with CUDA, it's time to move towards safer code.
This chapter will be mostly easier, but much more routine than previous ones.

## Conditional code
Unfortunately, we can't just blindly merge and forget both crates into one, because a **most** of the code needs to be compiled for an **only single target**, while only a small amount of the code needs to be shared.

We'll need to mark our code with either `#[cfg(target_os = "cuda")]` or `#[cfg(not(target_os = "cuda"))]` depending on is the code has to run on device side or a host side.

Sometimes it might be useful to specify a target for a whole module, like:
``` rust
#[cfg(not(target_os = "cuda"))]
mod static_cuda;
```

We can add the target guard to almost everything :)
``` rust
#[cfg(not(target_os = "cuda"))]
macro_rules! cuda_kernel { ... }

#[cfg(target_os = "cuda")]
pub use self::bilateral::bilateral_kernel;
```

Marking a code with target guards is takes significant time and efforts, but so far we can't avoid this.

### Crate dependencies
In the same way as normal code, we need to add target guards to our dependencies.
It has to be done from both `Cargo.toml` side and `src/lib.rs`.

In `Cargo.toml` we can specify separated dictionaries `[target.'cfg(...)'.dependencies]`:
``` toml
[package]
name = "chapter-2"

[target.'cfg(not(target_os = "cuda"))'.dependencies]
png = "0.7"

[target.'cfg(target_os = "cuda")'.dependencies]
nvptx-builtins = "0.1"
```

This will give us a `png` as host-side and `nvptx-builtins` as device-side dependency.

Also we need to specify guards for `extern crate ...;` in `src/lib.rs`:
``` rust
#[cfg(not(target_os = "cuda"))]
extern crate png;

#[cfg(target_os = "cuda")]
extern crate nvptx_builtins;
```

### Crate attributes
Quite important, to not forget about crate attributes.
We need at least couple for device-side, such as:
``` rust
#![feature(abi_ptx)]
#![no_std]
```

We must not use them on host-side, so we conditionally enable them:
``` rust
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", no_std)]
```

## Kernel arguments safety
Now we came to the most interesting point.
The goal of the chapter is to get a **safer code**.
We are going to check kernel arguments at compile time!

Let's say we have a kernel *placeholder* trait with `Args` where stored information about kernel arguments and method `get_name()` that provides kernel name:
``` rust
pub trait KernelPlaceholder {
    type Args;

    fn get_name() -> &'static str;
}
```

Normally it can be used like:
``` rust
struct some_kernel_placeholder;

impl KernelPlaceholder for some_kernel_placeholder {
    type Args = (*const Pixel, *mut Pixel, u32, f64, f64);

    fn get_name() -> &'static str {
        "some_kernel_name"
    }
}
```

Then we can define a kernel *wrapper* that holds a handle to CUDA kernel and the *placeholder* info:
``` rust
pub struct Kernel<F: KernelPlaceholder> {
    handle: Function<'static, 'static>,
    signature: PhantomData<F>,
}
```

Now let's create a trait with `execute` method for kernels with arity 5 (read, with 5 arguments) and implement it for our *wrapper*:
``` rust
pub trait ModuleKernelWithArity5<I1, I2, I3, I4, I5> {
    fn execute(
        &self,
        grid: Grid,
        block: Block,
        i1: I1,
        i2: I2,
        i3: I3,
        i4: I4,
        i5: I5,
    ) -> Result<(), driver::Error>;
}

impl<F, I1, I2, I3, I4, I5> ModuleKernelWithArity5<I1, I2, I3, I4, I5> for Kernel<F>
where
    F: KernelPlaceholder<Args = (I1, I2, I3, I4, I5)>,
{
    fn execute(
        &self,
        grid: Grid,
        block: Block,
        i1: I1,
        i2: I2,
        i3: I3,
        i4: I4,
        i5: I5,
    ) -> Result<(), driver::Error> {
        self.handle.launch(
            &[Any(&i1), Any(&i2), Any(&i3), Any(&i4), Any(&i5)],
            grid,
            block,
        )
    }
}
```

We will probably need to create as many traits as many different kernel arities we have.
Fortunately, we have an only single kernel with arity 5 in our example :)

It will be all for nothing if we make a mistake in specifying the kernel *placeholder* since it must have exactly the same arguments as our real kernel has.
We definitely need to automate this task!

### Kernel placeholders automatic generation
At the beginning of the chapter, we found out, that we can provide different macro implementations for each target.

``` rust
#[cfg(target_os = "cuda")]
macro_rules! cuda_kernel { ... host-side implementation ... }

#[cfg(not(target_os = "cuda"))]
macro_rules! cuda_kernel { ... device-side implementation ... }
```

For example, our kernel:
``` rust
cuda_kernel! {
    fn bilateral_kernel(
        src: *const Pixel,
        dst: *mut Pixel,
        radius: u32,
        sigma_d: f64,
        sigma_r: f64
    ) {
        self::device::bilateral_kernel(src, dst, radius, sigma_d, sigma_r);
    }
}
```

For device-side we need to expand the macro into a real kernel function:
``` rust
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bilateral_kernel(
    src: *const Pixel,
    dst: *mut Pixel,
    radius: u32,
    sigma_d: f64,
    sigma_r: f64
) {
    self::device::bilateral_kernel(src, dst, radius, sigma_d, sigma_r);
}
```

And for host-side we expand it into a *placeholder*:
``` rust
#[allow(non_camel_case_types)]
struct bilateral_kernel;

impl KernelPlaceholder for bilateral_kernel {
    type Args = (*const Pixel, *mut Pixel, u32, f64, f64);

    fn get_name() -> &'static str {
        "bilateral_kernel"
    }
}
```

The macro implementation can be found at [`host/src/lib.rs`](host/src/lib.rs).

If we accidentally pass wrong  into our kernel, we would end up with compilation errors!
```
error[E0308]: mismatched types
   --> chapter-2/host/src/filter/bilateral.rs:167:13
    |
167 |             sigma_d,
    |             ^^^^^^^ expected u32, found f64

error[E0308]: mismatched types
   --> chapter-2/host/src/filter/bilateral.rs:168:13
    |
168 |             radius as u32,
    |             ^^^^^^^^^^^^^ expected f64, found u32
```

## Results
Now we have both host and device code in one place, and we should not worry about passing wrong arguments for kernel invocation!
So far, wonderful results.
But, seriously, the final code lost all Rust's elegance and became just monstrous because of these many target guards!

Our next step would be creating a **proc-macro** helper to deal with kernel arguments type checking and with other target-conditional stuff.
