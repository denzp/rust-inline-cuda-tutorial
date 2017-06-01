#![feature(test)]

extern crate png;
extern crate clap;
extern crate rayon;
extern crate test;

use clap::{Arg, App};

mod image;
use image::Image;

mod filter;
use filter::bilateral_parallel;

fn main() {
    let matches = App::new("Bilateral Filter")
        .version("1.0")
        .author("Denys Zariaiev <denys.zariaiev@gmail.com>")
        .arg(Arg::with_name("INPUT")
            .help("Input PNG image path")
            .required(true)
            .index(1))
        .arg(Arg::with_name("OUTPUT")
            .help("Output PNG image path")
            .required(true)
            .index(2))
        .get_matches();

    let input_path = matches.value_of("INPUT").unwrap();
    let input_image = Image::open(input_path).unwrap();
    println!("Input image is {}x{} => {} pixels", input_image.width, input_image.height, input_image.pixels.len());

    let output_path = matches.value_of("OUTPUT").unwrap();
    let output_image = bilateral_parallel(&input_image, 5, 3.5, 3.0);
    println!("Output image is {}x{} => {} pixels", output_image.width, output_image.height, output_image.pixels.len());

    output_image.save(output_path).unwrap();
}