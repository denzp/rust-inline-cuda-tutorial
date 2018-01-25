#![feature(abi_ptx, intrinsics, lang_items)]
#![deny(warnings)]
#![no_std]

use core::cmp::{max, min};

extern crate nvptx_builtins;
use nvptx_builtins::*;

extern crate math;
use math::{exp, sqrt};

pub struct Pixel {
    r: u8,
    g: u8,
    b: u8,
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn bilateral_filter(
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

#[lang = "panic_fmt"]
fn panic_fmt() -> ! {
    loop {}
}
