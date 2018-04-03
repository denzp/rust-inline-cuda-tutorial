extern crate chapter_2;

mod utils;
use utils::compare_images;

use chapter_2::filter::bilateral_cuda as filter;
use chapter_2::image::Image;

#[test]
fn should_produce_correct_image_512() {
    let input = Image::open("../../fixtures/input-512.png").unwrap();

    let current_output = filter(&input, 5, 3.5, 3.0);
    let reference_output = Image::open("../../fixtures/ref-output-512.png").unwrap();

    compare_images(&current_output.unwrap(), &reference_output);
}

#[test]
fn should_produce_correct_image_1024() {
    let input = Image::open("../../fixtures/input-1024.png").unwrap();

    let current_output = filter(&input, 5, 3.5, 3.0);
    let reference_output = Image::open("../../fixtures/ref-output-1024.png").unwrap();

    compare_images(&current_output.unwrap(), &reference_output);
}
