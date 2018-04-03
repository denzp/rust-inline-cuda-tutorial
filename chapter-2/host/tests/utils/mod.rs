use chapter_2::image::Image;

pub fn compare_images(current: &Image, reference: &Image) {
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
