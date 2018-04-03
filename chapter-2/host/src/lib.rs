#![deny(warnings)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", no_std)]

#[macro_use]
#[cfg(not(target_os = "cuda"))]
extern crate lazy_static;

#[cfg(not(target_os = "cuda"))]
extern crate cuda;
#[cfg(not(target_os = "cuda"))]
extern crate png;

#[cfg(target_os = "cuda")]
extern crate math;
#[cfg(target_os = "cuda")]
extern crate nvptx_builtins;

#[cfg(not(target_os = "cuda"))]
mod static_cuda;

#[cfg(not(target_os = "cuda"))]
macro_rules! cuda_kernel {
    (
        fn
        $name:ident($arg0_name:ident : $arg0_type:ty $(, $arg_name:ident : $arg_type:ty)*)
        $body:block
    ) => {
        #[allow(non_camel_case_types)]
        struct $name;

        impl $crate::static_cuda::KernelPlaceholder for $name {
            type Args = ($arg0_type $(, $arg_type)*);

            fn get_name() -> &'static str {
                stringify!($name)
            }
        }
    };
}

#[cfg(target_os = "cuda")]
macro_rules! cuda_kernel {
    (fn $name:ident ($($arg:tt)*) $body:block) => {
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $name($($arg)*) $body
    };
}

pub mod filter;
pub mod image;
