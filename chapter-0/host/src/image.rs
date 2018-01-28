use std::fs::File;
use std::io::BufWriter;
use std::ptr;
use png::{Decoder, DecodingError, Encoder, EncodingError, ColorType, BitDepth, HasParameters};

pub struct Image {
    pub pixels: Vec<Pixel>,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Debug)]
pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Image {
            width,
            height,
            pixels: vec![Pixel::default(); width * height],
        }
    }

    pub fn open(path: &str) -> Result<Self, DecodingError> {
        let file = File::open(path)?;
        let decoder = Decoder::new(file);
        let (info, mut reader) = decoder.read_info()?;

        if info.color_type != ColorType::RGB {
            return Err(DecodingError::Other("Color type must be RGB!".into()));
        }

        if info.bit_depth != BitDepth::Eight {
            return Err(DecodingError::Other("Bit depth must be 8!".into()));
        }

        let mut buffer = vec![0; info.buffer_size()];
        let mut image = Image::new(info.width as usize, info.height as usize);

        reader.next_frame(&mut buffer)?;

        unsafe {
            ptr::copy_nonoverlapping(buffer.as_ptr(),
                                     image.pixels.as_mut_ptr() as *mut u8,
                                     buffer.len());

            Ok(image)
        }
    }

    pub fn save(&self, path: &str) -> Result<(), EncodingError> {
        let file = File::create(path)?;
        let mut encoder = Encoder::new(BufWriter::new(file), self.width as u32, self.height as u32);

        encoder.set(ColorType::RGB).set(BitDepth::Eight);

        let mut writer = encoder.write_header()?;
        let mut buffer = vec![0; self.pixels.len() * 3];

        unsafe {
            ptr::copy_nonoverlapping(self.pixels.as_ptr() as *const u8,
                                     buffer.as_mut_ptr(),
                                     buffer.len());
        }

        writer.write_image_data(&buffer)?;
        Ok(())
    }
}

impl Default for Pixel {
    fn default() -> Self {
        Pixel { r: 0, g: 0, b: 0 }
    }
}

impl PartialEq<Pixel> for Pixel {
    fn eq(&self, other: &Pixel) -> bool {
        self.r == other.r && self.g == other.g && self.b == other.b
    }
}

