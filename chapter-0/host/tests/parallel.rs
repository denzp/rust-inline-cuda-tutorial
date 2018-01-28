extern crate chapter_0;

mod utils;
use utils::compare_images;

use chapter_0::filter::bilateral_parallel as filter;
use chapter_0::image::Image;

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
