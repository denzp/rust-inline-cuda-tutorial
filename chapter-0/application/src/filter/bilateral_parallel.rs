use rayon::prelude::*;
use image::{Pixel, Image};
use std::cmp::{min, max};

pub fn filter(source: &Image, radius: usize, sigma_d: f64, sigma_r: f64) -> Image {
    let mut destination = Image::new(source.width, source.height);

    destination.pixels = (0..source.height * source.width)
        .into_par_iter()
        .map(|index| {
                 let i = index / source.width;
                 let j = index % source.width;

                 filter_pixel(source, radius, i, j, sigma_d, sigma_r)
             })
        .collect();


    destination
}

fn filter_pixel(source: &Image,
                radius: usize,
                i: usize,
                j: usize,
                sigma_d: f64,
                sigma_r: f64)
                -> Pixel {
    let mut r_value = 0f64;
    let mut r_accum = 0f64;

    let mut g_value = 0f64;
    let mut g_accum = 0f64;

    let mut b_value = 0f64;
    let mut b_accum = 0f64;

    for k in max(i as i32 - radius as i32, 0)..min(i as i32 + radius as i32, source.height as i32) {
        for l in max(j as i32 - radius as i32, 0)..
                 min(j as i32 + radius as i32, source.width as i32) {
            let w = w_kernel(i as i32, j as i32, k, l, source, sigma_d, sigma_r);

            r_value = r_value + w * source.pixels[k as usize * source.width + l as usize].r as f64;
            r_accum = r_accum + w;

            g_value = g_value + w * source.pixels[k as usize * source.width + l as usize].g as f64;
            g_accum = g_accum + w;

            b_value = b_value + w * source.pixels[k as usize * source.width + l as usize].b as f64;
            b_accum = b_accum + w;
        }
    }

    Pixel {
        r: (r_value / r_accum) as u8,
        g: (g_value / g_accum) as u8,
        b: (b_value / b_accum) as u8,
    }
}

fn w_kernel(i: i32, j: i32, k: i32, l: i32, source: &Image, sigma_d: f64, sigma_r: f64) -> f64 {
    let w_d = ((i - k) * (i - k) + (j - l) * (j - l)) as f64;
    let w_r = l2_distance(&source.pixels[i as usize * source.width + j as usize],
                          &source.pixels[k as usize * source.width + l as usize]);

    f64::exp(-w_d / (2.0 * sigma_d * sigma_d) - w_r / (2.0 * sigma_r * sigma_r))
}

fn l2_distance(lhs: &Pixel, rhs: &Pixel) -> f64 {
    let square = (lhs.r as f64 - rhs.r as f64) * (lhs.r as f64 - rhs.r as f64) +
                 (lhs.g as f64 - rhs.g as f64) * (lhs.g as f64 - rhs.g as f64) +
                 (lhs.b as f64 - rhs.b as f64) * (lhs.b as f64 - rhs.b as f64);

    square.sqrt()
}

#[cfg(test)]
mod tests {
    use super::filter;
    use image::Image;
    use test::Bencher;

    #[test]
    fn should_produce_correct_image_512() {
        let input = Image::open("../../fixtures/input-512.png").unwrap();

        let current_output = filter(&input, 5, 3.5, 3.0);
        let reference_output = Image::open("../../fixtures/ref-output-512.png").unwrap();

        compare_images(&current_output, &reference_output);
    }

    #[test]
    fn should_produce_correct_image_1024() {
        let input = Image::open("../../fixtures/input-1024.png").unwrap();

        let current_output = filter(&input, 5, 3.5, 3.0);
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

