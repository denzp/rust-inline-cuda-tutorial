use std::mem::size_of;
use cuda::driver;
use cuda::driver::{Any, Block, Direction, Error as CudaError, Grid};

use image::{Image, Pixel};
use static_cuda::{CUDA_CTX, CUDA_KERNEL};

pub fn filter(
    source: &Image,
    radius: usize,
    sigma_d: f64,
    sigma_r: f64,
) -> Result<Image, CudaError> {
    let mut destination = Image::new(source.width, source.height);

    CUDA_CTX.set_current()?;

    let d_src = unsafe { driver::allocate(source.pixels.len() * size_of::<Pixel>())? };
    let d_dst = unsafe { driver::allocate(destination.pixels.len() * size_of::<Pixel>())? };

    unsafe {
        driver::copy(
            source.pixels.as_ptr(),
            d_src as *mut Pixel,
            source.pixels.len(),
            Direction::HostToDevice,
        )?;
    }

    CUDA_KERNEL.launch(
        &[
            Any(&d_src),
            Any(&d_dst),
            Any(&(radius as u32)),
            Any(&sigma_d),
            Any(&sigma_r),
        ],
        Grid::xy(source.width as u32 / 8, source.height as u32 / 8),
        Block::xy(8, 8),
    )?;

    unsafe {
        driver::copy(
            d_dst as *mut Pixel,
            destination.pixels.as_mut_ptr(),
            destination.pixels.len(),
            Direction::DeviceToHost,
        )?;
        driver::deallocate(d_src)?;
        driver::deallocate(d_dst)?;
    }

    Ok(destination)
}
