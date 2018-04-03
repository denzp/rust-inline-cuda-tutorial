use image::Pixel;

cuda_kernel! {
    fn bilateral_kernel(src: *const Pixel, dst: *mut Pixel, radius: u32, sigma_d: f64, sigma_r: f64) {
        self::device::bilateral_kernel(src, dst, radius, sigma_d, sigma_r);
    }
}

#[cfg(target_os = "cuda")]
mod device {
    use core::cmp::{max, min};
    use image::Pixel;
    use math::{exp, sqrt};
    use nvptx_builtins::*;

    pub unsafe fn bilateral_kernel(
        src: *const Pixel,
        dst: *mut Pixel,
        radius: u32,
        sigma_d: f64,
        sigma_r: f64,
    ) {
        let width = grid_dim_x() * block_dim_x();
        let height = grid_dim_y() * block_dim_y();

        let j = (block_dim_x() * block_idx_x() + thread_idx_x()) as i32;
        let i = (block_dim_y() * block_idx_y() + thread_idx_y()) as i32;

        let src_image = Image {
            pixels: src,
            width: width as i32,
        };

        let mut r_value: f64 = 0.0;
        let mut r_accum: f64 = 0.0;

        let mut g_value: f64 = 0.0;
        let mut g_accum: f64 = 0.0;

        let mut b_value: f64 = 0.0;
        let mut b_accum: f64 = 0.0;

        for k in max(i - radius as i32, 0)..min(i + radius as i32, height as i32) {
            for l in max(j - radius as i32, 0)..min(j + radius as i32, width as i32) {
                let w = w_kernel(&src_image, i, j, k, l, sigma_d, sigma_r);

                r_value = r_value + w * src_image.pixel(k, l).r as f64;
                r_accum = r_accum + w;

                g_value = g_value + w * src_image.pixel(k, l).g as f64;
                g_accum = g_accum + w;

                b_value = b_value + w * src_image.pixel(k, l).b as f64;
                b_accum = b_accum + w;
            }
        }

        let mut dst_image = MutImage {
            pixels: dst,
            width: width as i32,
        };

        dst_image.mut_pixel(i, j).r = (r_value / r_accum) as u8;
        dst_image.mut_pixel(i, j).g = (g_value / g_accum) as u8;
        dst_image.mut_pixel(i, j).b = (b_value / b_accum) as u8;
    }

    struct Image {
        pixels: *const Pixel,
        width: i32,
    }

    struct MutImage {
        pixels: *mut Pixel,
        width: i32,
    }

    impl Image {
        fn offset(&self, i: i32, j: i32) -> isize {
            (i * self.width + j) as isize
        }

        unsafe fn pixel(&self, i: i32, j: i32) -> &Pixel {
            &*self.pixels.offset(self.offset(i, j))
        }
    }

    impl MutImage {
        fn offset(&self, i: i32, j: i32) -> isize {
            (i * self.width + j) as isize
        }

        unsafe fn mut_pixel(&mut self, i: i32, j: i32) -> &mut Pixel {
            &mut *self.pixels.offset(self.offset(i, j))
        }
    }

    unsafe fn w_kernel(
        source: &Image,
        i: i32,
        j: i32,
        k: i32,
        l: i32,
        sigma_d: f64,
        sigma_r: f64,
    ) -> f64 {
        let w_d = ((i - k) * (i - k) + (j - l) * (j - l)) as f64;
        let w_r = l2_distance(source.pixel(i, j), source.pixel(k, l));

        exp(-w_d / (2.0 * sigma_d * sigma_d) - w_r / (2.0 * sigma_r * sigma_r))
    }

    unsafe fn l2_distance(lhs: &Pixel, rhs: &Pixel) -> f64 {
        let r_distance = lhs.r as f64 - rhs.r as f64;
        let g_distance = lhs.g as f64 - rhs.g as f64;
        let b_distance = lhs.b as f64 - rhs.b as f64;

        sqrt(r_distance * r_distance + g_distance * g_distance + b_distance * b_distance)
    }
}

#[cfg(not(target_os = "cuda"))]
pub mod host {
    use cuda::driver;
    use cuda::driver::{Block, Direction, Error as CudaError, Grid};
    use std::mem::size_of;

    use image::{Image, Pixel};
    use static_cuda::prelude::*;
    use static_cuda::{CUDA_CTX, CUDA_MODULE};

    pub fn filter(
        source: &Image,
        radius: usize,
        sigma_d: f64,
        sigma_r: f64,
    ) -> Result<Image, CudaError> {
        let mut destination = Image::new(source.width, source.height);
        let kernel = CUDA_MODULE.kernel::<super::bilateral_kernel>()?;

        CUDA_CTX.set_current()?;

        let d_src = unsafe {
            let size = source.pixels.len() * size_of::<Pixel>();
            driver::allocate(size)? as *const Pixel
        };

        let d_dst = unsafe {
            let size = destination.pixels.len() * size_of::<Pixel>();
            driver::allocate(size)? as *mut Pixel
        };

        unsafe {
            driver::copy(
                source.pixels.as_ptr(),
                d_src as *mut Pixel,
                source.pixels.len(),
                Direction::HostToDevice,
            )?;
        }

        kernel.execute(
            Grid::xy(source.width as u32 / 8, source.height as u32 / 8),
            Block::xy(8, 8),
            d_src,
            d_dst,
            radius as u32,
            sigma_d,
            sigma_r,
        )?;

        unsafe {
            driver::copy(
                d_dst as *mut Pixel,
                destination.pixels.as_mut_ptr(),
                destination.pixels.len(),
                Direction::DeviceToHost,
            )?;

            driver::deallocate(d_src as *mut u8)?;
            driver::deallocate(d_dst as *mut u8)?;
        }

        Ok(destination)
    }
}
