use libwebp::WebPDecodeRGBA;
use png::Decoder;
use std::path::Path;
use turbojpeg::{Decompressor, Image, PixelFormat};

/// 画像デコードエラー型
#[derive(Debug)]
pub enum ImageDecodeError {
    Io(std::io::Error),
    Png(png::DecodingError),
    Jpeg(turbojpeg::Error),
    WebP(libwebp::error::WebPSimpleError),
    Image(image::ImageError),
    UnsupportedFormat,
}

impl From<std::io::Error> for ImageDecodeError {
    fn from(e: std::io::Error) -> Self {
        ImageDecodeError::Io(e)
    }
}
impl From<png::DecodingError> for ImageDecodeError {
    fn from(e: png::DecodingError) -> Self {
        ImageDecodeError::Png(e)
    }
}
impl From<turbojpeg::Error> for ImageDecodeError {
    fn from(e: turbojpeg::Error) -> Self {
        ImageDecodeError::Jpeg(e)
    }
}
impl From<libwebp::error::WebPSimpleError> for ImageDecodeError {
    fn from(e: libwebp::error::WebPSimpleError) -> Self {
        ImageDecodeError::WebP(e)
    }
}
impl From<image::ImageError> for ImageDecodeError {
    fn from(e: image::ImageError) -> Self {
        ImageDecodeError::Image(e)
    }
}

/// 指定パスの画像をRGBA8形式でデコードし、ピクセルバイト列と幅・高さを返す
pub fn decode_image_to_rgba8(path: &Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "png" => decode_png_to_rgba8(path),
        "jpg" | "jpeg" => decode_jpeg_to_rgba8(path),
        "webp" => decode_webp_to_rgba8(path),
        _ => Err(ImageDecodeError::UnsupportedFormat),
    }
}

/// pngクレートを使ったPNG画像のデコード
fn decode_png_to_rgba8(path: &Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
    let file = std::fs::File::open(path)?;
    let decoder = Decoder::new(file);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let width = info.width;
    let height = info.height;

    // RGBA8に変換
    let rgba = match info.color_type {
        png::ColorType::Rgba => {
            // 既にRGBA
            buf[..info.buffer_size()].to_vec()
        }
        png::ColorType::Rgb => {
            // RGB→RGBA
            let mut out = Vec::with_capacity(width as usize * height as usize * 4);
            let src = &buf[..info.buffer_size()];
            for chunk in src.chunks(3) {
                out.extend_from_slice(chunk);
                out.push(255);
            }
            out
        }
        png::ColorType::Grayscale => {
            // Gray→RGBA
            let mut out = Vec::with_capacity(width as usize * height as usize * 4);
            let src = &buf[..info.buffer_size()];
            for &g in src {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            out
        }
        png::ColorType::GrayscaleAlpha => {
            // GrayAlpha→RGBA
            let mut out = Vec::with_capacity(width as usize * height as usize * 4);
            let src = &buf[..info.buffer_size()];
            for chunk in src.chunks(2) {
                let g = chunk[0];
                let a = chunk[1];
                out.extend_from_slice(&[g, g, g, a]);
            }
            out
        }
        png::ColorType::Indexed => {
            // Indexedカラーはパレットを参照
            let palette = reader
                .info()
                .palette
                .as_ref()
                .ok_or(ImageDecodeError::UnsupportedFormat)?;
            let src = &buf[..info.buffer_size()];
            let trns = reader.info().trns.as_ref();
            let mut out = Vec::with_capacity(width as usize * height as usize * 4);
            for &idx in src {
                let i = idx as usize * 3;
                if i + 2 >= palette.len() {
                    return Err(ImageDecodeError::UnsupportedFormat);
                }
                let r = palette[i];
                let g = palette[i + 1];
                let b = palette[i + 2];
                let a = if let Some(trns) = trns {
                    if idx as usize >= trns.len() {
                        255
                    } else {
                        trns[idx as usize]
                    }
                } else {
                    255
                };
                out.extend_from_slice(&[r, g, b, a]);
            }
            out
        }
    };

    Ok((rgba, width, height))
}

/// turbojpegクレートを使ってJPEG画像のデコード
fn decode_jpeg_to_rgba8(path: &Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
    let data = std::fs::read(path)?;
    let mut decompressor = Decompressor::new()?;
    let header = decompressor.read_header(&data)?;

    let mut image = Image {
        pixels: vec![0; (header.width * header.height * 4) as usize],
        width: header.width,
        pitch: header.width * 4,
        height: header.height,
        format: PixelFormat::RGBA,
    };

    decompressor.decompress(&data, image.as_deref_mut())?;
    Ok((image.pixels, image.width as u32, image.height as u32))
}

/// libwebpクレートを使ってWebP画像のデコード
fn decode_webp_to_rgba8(path: &Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
    let data = std::fs::read(path)?;
    let (width, height, buf) = WebPDecodeRGBA(&data)?;
    Ok((buf.to_vec(), width, height))
}
