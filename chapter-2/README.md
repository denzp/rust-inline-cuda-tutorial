# Chapter 2
> Merging CPU and GPU code into one crate.

## Introduction
After first success with CUDA, it's time to add safety to our code.
This chapter will be mostly easier, but much more routine than previous ones.

Unfortunately, we can't just blindly merge and forget both crates into one, because a **most** of the code needs to be compiled for an **only single target**, while only a small amount of the code needs to be shared.

## Conditional code
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

In `Cargo.toml` we can write:
``` toml
[package]
name = "chapter-2"

[target.'cfg(not(target_os = "cuda"))'.dependencies]
png = "0.7"

[target.'cfg(target_os = "cuda")'.dependencies]
nvptx-builtins = "0.1"
```

And then we will have `png` as host-side and `nvptx-builtins` as device-side dependency.

Also we need to specify in `src/lib.rs`:
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

We need not use them on host-side, so we must conditionally enable them:
``` rust
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", no_std)]
```

## Kernel arguments safety
Now we came to the most interesting point.
The goal of the chapter is to get a **safer code**.
We can try to check kernel arguments at compile time!

Let's say we have a kernel wrapper
``` rust
pub trait KernelPlaceholder {
    type Args;

    fn get_name() -> &'static str;
}

pub struct Kernel<F: KernelPlaceholder> {
    kernel: Function<'static, 'static>,
    signature: PhantomData<F>,
}
```

Where `KernelPlaceholder::Args` stores information about kernel arguments and `KernelPlaceholder::get_name()` provides kernel name.

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

Then we might create a trait for kernels with arity 5 (read, with 5 arguments):
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
```

And now we can implement the trait for out kernel wrapper `Kernel<F>`:

``` rust
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
        self.kernel.launch(
            &[Any(&i1), Any(&i2), Any(&i3), Any(&i4), Any(&i5)],
            grid,
            block,
        )
    }
}
```

We will need to create as many traits as many different kernel arities we have.
Fortunately, we have an only single kernel in our example :)

It will be all for nothing if we make a mistake in specifying the kernel descriptor since it must have exactly the same arguments as our kernel has.
It would be great if we can automatically create it!

(TODO: funny note about macros)

(TODO: link to the macro code)

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

We are able to produce code for device-side:
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

And different for host-side:
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

(TODO: link to code that was changed to get the error)
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
But, seriously, the final code became just monstrous (TODO)!

Our next step would be creating a **proc-macro** helper to deal with kernel arguments type checking and with other target-conditional stuff.
