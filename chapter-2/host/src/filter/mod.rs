mod bilateral;

#[cfg(target_os = "cuda")]
pub use self::bilateral::bilateral_kernel;

#[cfg(not(target_os = "cuda"))]
pub use self::bilateral::host::filter as bilateral_cuda;
