#![feature(test)]
#![deny(warnings)]

#[macro_use]
extern crate lazy_static;

extern crate clap;
extern crate cuda;
extern crate png;
extern crate test;

use clap::{App, Arg};

mod image;
use image::Image;

mod static_cuda;
mod filter;
use filter::bilateral_cuda;

fn main() {
    let matches = App::new("Bilateral Filter")
        .version("1.0")
        .author("Denys Zariaiev <denys.zariaiev@gmail.com>")
        .arg(
            Arg::with_name("INPUT")
                .help("Input PNG image path")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("OUTPUT")
                .help("Output PNG image path")
                .required(true)
                .index(2),
        )
        .get_matches();

    let input_path = matches.value_of("INPUT").unwrap();
    let input_image = Image::open(input_path).unwrap();

    println!(
        "Input image is {}x{} => {} pixels",
        input_image.width,
        input_image.height,
        input_image.pixels.len()
    );

    let output_path = matches.value_of("OUTPUT").unwrap();
    let output_image =
        bilateral_cuda(&input_image, 5, 3.5, 3.0).expect("Unable to filter an image");

    println!(
        "Output image is {}x{} => {} pixels",
        output_image.width,
        output_image.height,
        output_image.pixels.len()
    );

    output_image.save(output_path).unwrap();
}
