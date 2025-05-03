use eframe::{
    NativeOptions,
    egui::{self, TextureId},
    wgpu::{
        self, AddressMode, Device, Extent3d, FilterMode, Origin3d, Queue, Sampler,
        SamplerDescriptor, Texture, TextureAspect, TextureDescriptor, TextureDimension,
        TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    },
};
use egui::Ui;
use image::{ImageError, flat::View};
use libwebp::{WebPDecodeRGBA, WebPGetInfo, error::WebPSimpleError as WebPError};
use num_cpus;
use png::{BitDepth, ColorType, Decoder, Encoder};
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use rfd::FileDialog;
use std::sync::mpsc::{self, Receiver, Sender};
// use std::sync::mpsc::{Receiver, Sender, channel};
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    fmt,
    fs::{self, File},
    io::BufReader,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};
use turbojpeg::{Decompressor, Image, PixelFormat, Subsamp, compress_image, decompress_image};
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}
pub struct GpuImageRequest {
    pub path: PathBuf,
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct MyApp {
    folder: Option<PathBuf>,
    image_files: Vec<PathBuf>,
    current_index: usize,
    previous_index: usize,
    priority_list: Vec<PathBuf>,
    // 追加: デコード済み画像データ受信用
    decoded_rx: Option<Receiver<GpuImageRequest>>,
    decoded_tx: Option<Sender<GpuImageRequest>>,
    // 追加: GPUテクスチャキャッシュ
    texture_cache: HashMap<PathBuf, TextureId>,
    thread_pool: Option<Arc<ThreadPool>>,
}

impl Default for MyApp {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        let num_threads = num_cpus::get();
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        Self {
            folder: None,
            image_files: Vec::new(),
            current_index: 0,
            previous_index: usize::MAX,
            priority_list: Vec::new(),
            decoded_rx: Some(rx),
            decoded_tx: Some(tx),
            texture_cache: HashMap::new(),
            thread_pool: Some(Arc::new(pool)),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // メインスレッドでデコード済み画像を受信し、GPUに登録
        if let Some(rx) = &self.decoded_rx {
            while let Ok(req) = rx.try_recv() {
                // すでにキャッシュ済みならスキップ
                if self.texture_cache.contains_key(&req.path) {
                    continue;
                }
                // GPU登録
                if let Some((device, queue)) = frame
                    .wgpu_render_state()
                    .map(|r| (r.device.clone(), r.queue.clone()))
                {
                    let (_tex, view, sampler) = create_gpu_texture_from_rgba8_aligned(
                        std::sync::Arc::new(device),
                        std::sync::Arc::new(queue),
                        &req.pixels,
                        req.width,
                        req.height,
                    );
                    // register_wgpu_texture_to_eguiを使ってTextureIdを取得
                    let tex_id = register_wgpu_texture_to_egui(frame, &view, &sampler);
                    self.texture_cache.insert(req.path.clone(), tex_id);

                    //print
                    println!(
                        "Registered texture: {:?} (width: {}, height: {})",
                        req.path, req.width, req.height
                    );
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            folder_dialog(self, ui);
            navigation_buttons(self, ui);
            if !self.image_files.is_empty() && self.current_index != self.previous_index {
                on_image_index_changed(self, frame, ctx);
                self.previous_index = self.current_index;
            }
            // TextureIdから画像の表示
            if !self.image_files.is_empty() {
                let current_path = &self.image_files[self.current_index];
                if let Some(&tex_id) = self.texture_cache.get(current_path) {
                    // 仮に512x512で表示
                    let size = egui::Vec2::new(512.0, 512.0);
                    ui.image((tex_id, size));
                } else {
                    ui.label("Loading...");
                }
            }
        });
    }
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
            texture_view,
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

/// GPUからテクスチャを明示的に解放する（最適化版）
fn unregister_wgpu_texture(app: &mut MyApp, ctx: &egui::Context) {
    // priority_listに含まれない画像のTextureIdのみ削除
    let priority_set: std::collections::HashSet<_> = app.priority_list.iter().collect();

    // retainで不要なものだけ削除しつつ、egui側も解放
    let tex_manager = ctx.tex_manager();
    let mut tex_manager = tex_manager.write();

    // 削除対象のPathBufを集めてから削除
    let to_remove: Vec<_> = app
        .texture_cache
        .iter()
        .filter(|(path, _)| !priority_set.contains(path))
        .map(|(path, _)| path.clone())
        .collect();

    for path in to_remove {
        if let Some(tex_id) = app.texture_cache.remove(&path) {
            tex_manager.free(tex_id);
        }
    }
}

// prefetch_tasks_multipoolの書き換え
fn prefetch_tasks_multipool(app: &mut MyApp) {
    if app.priority_list.is_empty() {
        return;
    }
    let pool = app.thread_pool.as_ref().unwrap().clone();
    let tx = app.decoded_tx.as_ref().unwrap().clone();

    for path in app.priority_list.iter() {
        if app.texture_cache.contains_key(path) {
            continue;
        }
        let path = path.clone();
        let tx = tx.clone();
        pool.spawn_fifo(move || match decode_image_to_rgba8(&path) {
            Ok((pixels, width, height)) => {
                let _ = tx.send(GpuImageRequest {
                    path: path.clone(),
                    pixels,
                    width,
                    height,
                });
                println!("Decoded: {:?}", path);
            }
            Err(e) => {
                eprintln!("Failed to decode {:?}: {:?}", path, e);
            }
        });
    }
}

/// RGBA8画像データをGPUに転送しテクスチャを作成する（パディング対応）
fn create_gpu_texture_from_rgba8_aligned(
    device: Arc<Device>,
    queue: Arc<Queue>,
    rgba_pixels: &[u8],
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let bytes_per_pixel = 4;
    let unpadded_bytes_per_row = width as usize * bytes_per_pixel;
    let align = 256;
    let padding = (align - (unpadded_bytes_per_row % align)) % align;
    let padded_bytes_per_row = unpadded_bytes_per_row + padding;

    let mut padded_data = vec![0u8; padded_bytes_per_row * height as usize];
    for y in 0..height as usize {
        let src_start = y * unpadded_bytes_per_row;
        let src_end = src_start + unpadded_bytes_per_row;
        let dst_start = y * padded_bytes_per_row;
        let dst_end = dst_start + unpadded_bytes_per_row;
        padded_data[dst_start..dst_end].copy_from_slice(&rgba_pixels[src_start..src_end]);
    }

    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ImageTexture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &padded_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(padded_bytes_per_row as u32),
            rows_per_image: Some(height),
        },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    (texture, view, sampler)
}

fn on_image_index_changed(app: &mut MyApp, frame: &mut eframe::Frame, ctx: &egui::Context) {
    if app.image_files.is_empty() {
        return;
    }
    build_prefetch_list(app);
    // unregister_wgpu_texture(app, ctx);
    println!("Prefetching images: {:?}", app.priority_list);
    prefetch_tasks_multipool(app);
}
fn build_prefetch_list(app: &mut MyApp) {
    let total = app.image_files.len();
    if total == 0 {
        app.priority_list.clear();
        return;
    }
    let center = app.current_index;
    let mut paths = Vec::new();

    // 前後5枚＋中心画像で最大11枚
    // 近い順（中心→前→後→前→後...）で追加
    paths.push(app.image_files[center].clone());

    for offset in 1..=5 {
        // 前
        let idx = if let Some(idx) = center.checked_sub(offset) {
            idx
        } else {
            total - offset
        };
        paths.push(app.image_files[idx].clone());
        // 後
        let idx = center + offset;
        if idx < total {
            paths.push(app.image_files[idx].clone());
        }
    }

    // 近い順に並べる（中心→中心-1→中心+1→中心-2→中心+2...）
    app.priority_list = paths;
}

// フォルダ選択
fn folder_dialog(app: &mut MyApp, ui: &mut Ui) {
    if ui.button("Open Folder").clicked() {
        if let Some(folder) = FileDialog::new().pick_folder() {
            app.folder = Some(folder.clone());
            app.image_files = get_image_files(folder.to_str().unwrap());
        }
    }
}

// 画像ファイルの取得
fn get_image_files(dir: &str) -> Vec<PathBuf> {
    let mut image_files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
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

// 次、前 の画像を表示するためのボタン
fn navigation_buttons(app: &mut MyApp, ui: &mut Ui) {
    if !app.image_files.is_empty() {
        ui.horizontal(|ui| {
            if ui.button("Previous").clicked() {
                if app.current_index > 0 {
                    app.current_index -= 1;
                }
            }
            if ui.button("Next").clicked() {
                if app.current_index < app.image_files.len() - 1 {
                    app.current_index += 1;
                }
            }
        });
    }
}

/// 画像デコードエラー型
#[derive(Debug)]
pub enum ImageDecodeError {
    Io(std::io::Error),
    Png(png::DecodingError),
    Jpeg(turbojpeg::Error),
    WebP(libwebp::error::WebPSimpleError),
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

/// 指定パスの画像をRGBA8形式でデコードし、ピクセルバイト列と幅・高さを返す
fn decode_image_to_rgba8(path: &Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
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

/// PNG画像のデコード
fn decode_png_to_rgba8(path: &Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
    let file = File::open(path)?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());
    Ok((buf, info.width, info.height))
}

/// JPEG画像のデコード
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

/// WebP画像のデコード
fn decode_webp_to_rgba8(path: &Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
    let data = std::fs::read(path)?;
    let (width, height, buf) = WebPDecodeRGBA(&data)?;
    Ok((buf.to_vec(), width, height))
}
