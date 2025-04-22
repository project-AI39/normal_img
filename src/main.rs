// egui/eframe 関連
use eframe::{NativeOptions, egui, wgpu};
use egui::{ColorImage, Context, TextureHandle, TextureId, Ui};

// 画像処理
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageError};

// ファイルパスやファイル操作
use std::fs;
use std::path::PathBuf;

// wgpu 関連
use wgpu::{
    AddressMode, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, Extent3d,
    FilterMode, Origin3d, Queue, Sampler, SamplerDescriptor, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

// その他
use rfd::FileDialog;

pub struct MyApp {
    selected_folder: Option<String>,
    image_files: Vec<PathBuf>,
    current_index: usize,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            selected_folder: None,
            image_files: Vec::new(),
            current_index: 0,
        }
    }
}

// 画像ファイル一覧取得関数
fn get_image_files(dir: &str) -> Vec<PathBuf> {
    let mut image_files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "webp" {
                        image_files.push(path);
                    }
                }
            }
        }
    }
    image_files
}

// 指定パスの画像をRGBA8形式でデコードし、バイト列と幅・高さを返す
fn decode_image_to_rgba8(path: &std::path::Path) -> Result<(Vec<u8>, u32, u32), ImageError> {
    // 画像ファイルを開く
    let img = image::open(path)?;
    // RGBA8形式に変換
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    // バイト列として取り出す
    let bytes = rgba.into_raw();
    Ok((bytes, width, height))
}

/// RGBA8画像データをGPUに転送しテクスチャを作成する（アライメント対応）
fn create_gpu_texture_from_rgba8_aligned(
    device: &Device,
    queue: &Queue,
    rgba_pixels: &[u8],
    width: u32,
    height: u32,
) -> (Texture, TextureView, Sampler) {
    let bytes_per_pixel = 4;
    let unpadded_bytes_per_row = width as usize * bytes_per_pixel;
    let align = 256;
    let padding = (align - (unpadded_bytes_per_row % align)) % align;
    let padded_bytes_per_row = unpadded_bytes_per_row + padding;

    // パディングを加えたバッファを作成
    let mut padded_data = vec![0u8; padded_bytes_per_row * height as usize];
    for y in 0..height as usize {
        let src_start = y * unpadded_bytes_per_row;
        let src_end = src_start + unpadded_bytes_per_row;
        let dst_start = y * padded_bytes_per_row;
        let dst_end = dst_start + unpadded_bytes_per_row;
        padded_data[dst_start..dst_end].copy_from_slice(&rgba_pixels[src_start..src_end]);
        // パディング部分（dst_end..dst_start+pad）はゼロのままでOK
    }

    // テクスチャ作成
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("ImageTexture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // GPUへ画像データ転送
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &padded_data,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(padded_bytes_per_row as u32),
            rows_per_image: Some(height),
        },
        size,
    );

    // テクスチャビュー作成
    let view = texture.create_view(&TextureViewDescriptor::default());

    // サンプラー作成
    let sampler = device.create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Nearest,
        ..Default::default()
    });

    (texture, view, sampler)
}

fn register_wgpu_texture_to_egui(
    frame: &mut eframe::Frame,
    texture_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> egui::TextureId {
    // eframe::Frameからwgpuのレンダラを取得
    if let Some(render_state) = frame.wgpu_render_state() {
        let mut renderer = render_state.renderer.write();
        renderer.register_native_texture_with_sampler_options(
            &render_state.device,
            &texture_view,
            wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            },
        )
    } else {
        panic!("wgpu_render_state not available");
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if ui.button("Select Folder").clicked() {
                if let Some(folder) = FileDialog::new().pick_folder() {
                    let folder_str = folder.display().to_string();
                    self.selected_folder = Some(folder_str.clone());
                    self.image_files = get_image_files(&folder_str);

                    // Output to log
                    for img in &self.image_files {
                        println!("Image file: {}", img.display());
                    }
                }
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = NativeOptions {
        renderer: eframe::Renderer::Wgpu, // wgpuレンダラーを指定
        ..Default::default()
    };
    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}
