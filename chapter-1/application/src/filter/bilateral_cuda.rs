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

#[cfg(test)]
mod tests {
    use super::filter;
    use image::Image;
    use test::Bencher;

    #[test]
    fn should_produce_correct_image_512() {
        let input = Image::open("../../fixtures/input-512.png").unwrap();

        let current_output = filter(&input, 5, 3.5, 3.0).unwrap();
        let reference_output = Image::open("../../fixtures/ref-output-512.png").unwrap();

        compare_images(&current_output, &reference_output);
    }

    #[test]
    fn should_produce_correct_image_1024() {
        let input = Image::open("../../fixtures/input-1024.png").unwrap();

        let current_output = filter(&input, 5, 3.5, 3.0).unwrap();
        let reference_output = Image::open("../../fixtures/ref-output-1024.png").unwrap();

        compare_images(&current_output, &reference_output);
    }

    #[bench]
    fn bench_512(b: &mut Bencher) {
        let input = Image::open("../../fixtures/input-512.png").unwrap();

        b.iter(|| filter(&input, 5, 3.5, 3.0))
    }

    #[bench]
    fn bench_1024(b: &mut Bencher) {
        let input = Image::open("../../fixtures/input-1024.png").unwrap();

        b.iter(|| filter(&input, 5, 3.5, 3.0))
    }

    #[bench]
    fn bench_2048(b: &mut Bencher) {
        let input = Image::open("../../fixtures/input-2048.png").unwrap();

        b.iter(|| filter(&input, 5, 3.5, 3.0))
    }

    #[bench]
    fn bench_4096(b: &mut Bencher) {
        let input = Image::open("../../fixtures/input-4096.png").unwrap();

        b.iter(|| filter(&input, 5, 3.5, 3.0))
    }

    fn compare_images(current: &Image, reference: &Image) {
        assert_eq!(current.width, reference.width);
        assert_eq!(current.height, reference.height);
        assert_eq!(current.pixels.len(), reference.pixels.len());

        let mut defferent_pixels_count = 0;
        for index in 0..current.pixels.len() {
            if current.pixels[index] != reference.pixels[index] {
                defferent_pixels_count += 1;
            }
        }

        assert_eq!(defferent_pixels_count, 0);
    }
}
