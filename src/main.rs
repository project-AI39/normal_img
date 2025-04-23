use eframe::{NativeOptions, egui, wgpu};
use egui::TextureId;
use image::ImageError;
use rfd::FileDialog;
use std::fs;
use std::path::PathBuf;
use wgpu::{
    AddressMode, Device, Extent3d, FilterMode, Origin3d, Queue, Sampler, SamplerDescriptor,
    Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

pub struct MyApp {
    selected_folder: Option<String>,
    image_files: Vec<PathBuf>,
    current_index: usize,
    current_texture_id: Option<egui::TextureId>,
    current_size: Option<(u32, u32)>,
    prev_index: usize,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            selected_folder: None,
            image_files: Vec::new(),
            current_index: 0,
            current_texture_id: None,
            current_size: None,
            prev_index: usize::MAX, // 強制的に初回ロード
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
    let img = image::open(path)?;
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
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
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &padded_data,
        wgpu::TexelCopyBufferLayout {
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

fn unregister_wgpu_texture_from_egui(frame: &mut eframe::Frame, texture_id: egui::TextureId) {
    if let Some(render_state) = frame.wgpu_render_state() {
        let mut renderer = render_state.renderer.write();
        renderer.free_texture(&texture_id);
    }
}

/// 画像パスを指定して、古いテクスチャを解放しつつ新しいテクスチャを登録し、TextureIdと画像サイズを返す
fn load_and_register_texture(
    frame: &mut eframe::Frame,
    path: &PathBuf,
    prev_texture_id: Option<egui::TextureId>,
) -> Option<(egui::TextureId, (u32, u32))> {
    // 古いテクスチャの解放
    if let Some(texture_id) = prev_texture_id {
        unregister_wgpu_texture_from_egui(frame, texture_id);
    }

    // 画像デコード
    let (pixels, width, height) = match decode_image_to_rgba8(path) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Failed to decode image: {}", e);
            return None;
        }
    };

    // wgpuデバイス・キュー取得
    let render_state = match frame.wgpu_render_state() {
        Some(rs) => rs,
        None => {
            eprintln!("wgpu_render_state not available");
            return None;
        }
    };
    let device = &render_state.device;
    let queue = &render_state.queue;

    // テクスチャ作成
    let (_texture, view, sampler) =
        create_gpu_texture_from_rgba8_aligned(device, queue, &pixels, width, height);

    // eguiにテクスチャ登録
    let texture_id = register_wgpu_texture_to_egui(frame, &view, &sampler);

    Some((texture_id, (width, height)))
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Drop用にframeを保存（unsafe: drop時に使うため）

        egui::CentralPanel::default().show(ctx, |ui| {
            // フォルダ選択
            if ui.button("Select Folder").clicked() {
                if let Some(folder) = FileDialog::new().pick_folder() {
                    let folder_str = folder.display().to_string();
                    self.selected_folder = Some(folder_str.clone());
                    self.image_files = get_image_files(&folder_str);
                    self.current_index = 0;
                    self.prev_index = usize::MAX; // 強制リロード
                }
            }

            // 画像送りUI
            if !self.image_files.is_empty() {
                ui.horizontal(|ui| {
                    if ui.button("Previous").clicked() && self.current_index > 0 {
                        self.current_index -= 1;
                    }
                    ui.label(format!(
                        "Image {}/{}",
                        self.current_index + 1,
                        self.image_files.len()
                    ));
                    if ui.button("Next").clicked()
                        && self.current_index < self.image_files.len() - 1
                    {
                        self.current_index += 1;
                    }
                });
            }

            // 画像切り替え時のみロード＆登録
            if !self.image_files.is_empty() && self.current_index != self.prev_index {
                if let Some(path) = self.image_files.get(self.current_index) {
                    let result =
                        load_and_register_texture(frame, path, self.current_texture_id.take());
                    if let Some((texture_id, size)) = result {
                        self.current_texture_id = Some(texture_id);
                        self.current_size = Some(size);
                    } else {
                        self.current_texture_id = None;
                        self.current_size = None;
                    }
                }
                self.prev_index = self.current_index;
            }

            // 画像表示
            if let (Some(texture_id), Some((width, height))) =
                (self.current_texture_id, self.current_size)
            {
                ui.image((texture_id, egui::Vec2::new(width as f32, height as f32)));
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}
