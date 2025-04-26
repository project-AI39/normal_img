use eframe::{
    NativeOptions, egui,
    wgpu::{
        self,
        core::device::{self, queue},
    },
};
use egui::TextureId;
use image::{ImageError, flat::View};
use num_cpus;
use rfd::FileDialog;
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    fmt, thread,
};
use std::{fs, sync::Arc};
use std::{fs::File, sync::atomic::AtomicBool};
use std::{io::BufReader, sync::atomic::Ordering};
use std::{path::PathBuf, sync::Mutex};
use wgpu::{
    AddressMode, Device, Extent3d, FilterMode, Origin3d, Queue, Sampler, SamplerDescriptor,
    Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};
// 画像デコードライブラリ
use libwebp::WebPDecodeRGBA;
use libwebp::error::WebPSimpleError as WebPError;
use png::{BitDepth, ColorType, Decoder, Encoder};
use turbojpeg::{Decompressor, Image, PixelFormat, Subsamp, compress_image, decompress_image};

macro_rules! meas {
    ($x:expr) => {{
        let start = std::time::Instant::now();
        let result = $x;
        let end = start.elapsed();
        let total_ms = end.as_secs() * 1000 + end.subsec_millis() as u64;
        println!(
            "{} Elapsed: {}ms ({}s{}ms)",
            stringify!($x),
            total_ms,
            end.as_secs(),
            end.subsec_millis()
        );
        result
    }};
}
struct PrefetchTask {
    index: usize,
    priority: usize, // 近いほど小さい
    is_cached: bool,
}
// タスクの優先度比較（BinaryHeapでpriority小さい順）
impl Ord for PrefetchTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.priority.cmp(&self.priority) // 小さいpriorityが優先
    }
}
impl PartialOrd for PrefetchTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for PrefetchTask {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl Eq for PrefetchTask {}

struct VramCacheEntry {
    texture_id: egui::TextureId,
    size_bytes: usize,
    width: u32,
    height: u32,
}

struct PrefetchTaskHandle {
    index: usize,
    cancel_flag: Arc<AtomicBool>,
}

impl PrefetchTaskHandle {
    fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }
    fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::SeqCst)
    }
}
// ImageDecodeErrorのDisplayトレイト実装
impl fmt::Display for ImageDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageDecodeError::Io(e) => write!(f, "I/Oエラー: {}", e),
            ImageDecodeError::Png(e) => write!(f, "PNGデコードエラー: {}", e),
            ImageDecodeError::Jpeg(e) => write!(f, "JPEGデコードエラー: {}", e),
            ImageDecodeError::WebP(e) => write!(f, "WebPデコードエラー: {:?}", e),
            ImageDecodeError::UnsupportedFormat => write!(f, "サポートされていない画像形式"),
        }
    }
}

#[derive(Debug)]
pub enum ImageDecodeError {
    Io(std::io::Error),
    Png(png::DecodingError),
    Jpeg(turbojpeg::Error),
    WebP(libwebp::error::WebPSimpleError),
    UnsupportedFormat,
}

impl From<std::io::Error> for ImageDecodeError {
    fn from(err: std::io::Error) -> Self {
        ImageDecodeError::Io(err)
    }
}

impl From<png::DecodingError> for ImageDecodeError {
    fn from(e: png::DecodingError) -> Self {
        ImageDecodeError::Png(e)
    }
}

impl From<turbojpeg::Error> for ImageDecodeError {
    fn from(err: turbojpeg::Error) -> Self {
        ImageDecodeError::Jpeg(err)
    }
}

impl From<libwebp::error::WebPSimpleError> for ImageDecodeError {
    fn from(e: libwebp::error::WebPSimpleError) -> Self {
        ImageDecodeError::WebP(e)
    }
}

pub struct MyApp {
    selected_folder: Option<String>,
    image_files: Vec<PathBuf>,
    current_index: usize,
    current_texture_id: Option<egui::TextureId>,
    current_size: Option<(u32, u32)>,
    prev_index: usize,
    vram_cache: HashMap<usize, VramCacheEntry>,
    vram_cache_total_bytes: usize,
    vram_limit_bytes: usize,
    prefetch_tasks: Vec<PrefetchTaskHandle>,
    prefetch_queue: Arc<Mutex<BinaryHeap<PrefetchTask>>>,
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
            vram_cache: HashMap::new(),
            vram_cache_total_bytes: 0,
            vram_limit_bytes: 512 * 1024 * 1024, // 例: 512MB
            prefetch_tasks: Vec::new(),
            prefetch_queue: Arc::new(Mutex::new(BinaryHeap::new())),
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

fn on_image_index_changed(app: &mut MyApp) {
    vram_cache_check(app);
    let prefetch_list = build_prefetch_list(app);
    release_unused_vram_images(&prefetch_list, app);
    cleanup_old_prefetch_tasks(&prefetch_list, app);
    start_prefetch_tasks(&prefetch_list, app);
}
fn vram_cache_check(app: &mut MyApp) {
    app.vram_cache_total_bytes = app.vram_cache.values().map(|e| e.size_bytes).sum();
}
fn get_image_size_bytes(path: &std::path::Path) -> Option<usize> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    let (width, height) = match ext.as_str() {
        "png" => {
            let file = File::open(path).ok()?;
            let decoder = png::Decoder::new(BufReader::new(file));
            let mut reader = decoder.read_info().ok()?;
            let info = reader.info();
            (info.width, info.height)
        }
        "jpg" | "jpeg" => {
            let data = std::fs::read(path).ok()?;
            let mut decompressor = turbojpeg::Decompressor::new().ok()?;
            let header = decompressor.read_header(&data).ok()?;
            (header.width as u32, header.height as u32)
        }
        "webp" => {
            let data = std::fs::read(path).ok()?;
            let (width, height, _) = libwebp::WebPDecodeRGBA(&data).ok()?;
            (width, height)
        }
        _ => return None,
    };

    let bytes_per_pixel = 4;
    let unpadded_bytes_per_row = width as usize * bytes_per_pixel;
    let align = 256;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
    let total_bytes = padded_bytes_per_row * height as usize;
    Some(total_bytes)
}
fn build_prefetch_list(app: &MyApp) -> Vec<PrefetchTask> {
    let mut list = Vec::new();
    let mut used_bytes = app.vram_cache_total_bytes;
    let vram_limit = app.vram_limit_bytes;
    let mut indices: Vec<usize> = (0..app.image_files.len()).collect();
    indices.sort_by_key(|&i| (i as isize - app.current_index as isize).abs());
    for &idx in &indices {
        if app.vram_cache.contains_key(&idx) {
            list.push(PrefetchTask {
                index: idx,
                priority: (idx as isize - app.current_index as isize).abs() as usize,
                is_cached: true,
            });
            continue;
        }
        // Option<usize> なので matchやunwrap_orで処理
        let size_bytes_opt = get_image_size_bytes(&app.image_files[idx]);
        let size_bytes = match size_bytes_opt {
            Some(sz) => sz,
            None => continue, // サイズ取得失敗時はスキップ
        };
        if used_bytes + size_bytes > vram_limit {
            break;
        }
        used_bytes += size_bytes;
        list.push(PrefetchTask {
            index: idx,
            priority: (idx as isize - app.current_index as isize).abs() as usize,
            is_cached: false,
        });
    }
    list
}

fn release_unused_vram_images(prefetch_list: &[PrefetchTask], app: &mut MyApp) {
    let prefetch_indices: HashSet<_> = prefetch_list.iter().map(|t| t.index).collect();
    let to_remove: Vec<_> = app
        .vram_cache
        .keys()
        .filter(|idx| !prefetch_indices.contains(idx))
        .cloned()
        .collect();
    for idx in to_remove {
        if let Some(entry) = app.vram_cache.remove(&idx) {
            // eguiテクスチャも解放
            // unregister_wgpu_texture_from_egui(frame, entry.texture_id); // frameはupdate()内で利用
        }
    }
}
fn cleanup_old_prefetch_tasks(prefetch_list: &[PrefetchTask], app: &mut MyApp) {
    let prefetch_indices: HashSet<_> = prefetch_list.iter().map(|t| t.index).collect();
    for task in app.prefetch_tasks.iter_mut() {
        if !prefetch_indices.contains(&task.index) {
            task.cancel();
        }
    }
    app.prefetch_tasks.retain(|task| !task.is_cancelled());
}
fn start_prefetch_tasks(prefetch_list: &[PrefetchTask], app: &mut MyApp) {
    let num_workers = num_cpus::get();
    let mut active_tasks = 0;
    for task in prefetch_list {
        if task.is_cached {
            continue;
        }
        if active_tasks >= num_workers {
            break;
        }
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let image_files = app.image_files.clone();
        let idx = task.index;
        let queue = app.prefetch_queue.clone();
        // 必要に応じてwgpu device/queue, frame, vram_cache等をArc/Mutexで渡す
        // 実際はクロージャ内でGPU転送・egui登録・vram_cache登録
        thread::spawn({
            let cancel_flag = cancel_flag.clone();
            move || {
                if cancel_flag.load(Ordering::SeqCst) {
                    return;
                }
                // 画像デコード
                let (pixels, width, height) = match decode_image_to_rgba8(&image_files[idx]) {
                    Ok(res) => res,
                    Err(_) => return,
                };
                if cancel_flag.load(Ordering::SeqCst) {
                    return;
                }
                // GPU転送・egui登録・vram_cache登録（省略: frame, device, queue等が必要）
                // ...
            }
        });
        app.prefetch_tasks.push(PrefetchTaskHandle {
            index: idx,
            cancel_flag,
        });
        active_tasks += 1;
    }
}
// 指定パスの画像をRGBA8形式でデコードし、バイト列と幅・高さを返す
// fn decode_image_to_rgba8(path: &std::path::Path) -> Result<(Vec<u8>, u32, u32), ImageError> {
//     let img = image::open(path)?;
//     let rgba = img.to_rgba8();
//     let (width, height) = rgba.dimensions();
//     let bytes = rgba.into_raw();
//     Ok((bytes, width, height))
// }
fn decode_image_to_rgba8(path: &std::path::Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
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

fn decode_png_to_rgba8(path: &std::path::Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
    let file = File::open(path)?;
    let decoder = Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());
    Ok((buf, info.width, info.height))
}

fn decode_jpeg_to_rgba8(path: &std::path::Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
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

// WebPデコード関数
fn decode_webp_to_rgba8(path: &std::path::Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
    let data = std::fs::read(path)?;
    let (width, height, buf) = libwebp::WebPDecodeRGBA(&data)?;
    Ok((buf.to_vec(), width, height)) // to_vec() で変換
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
    meas!({
        if let Some(texture_id) = prev_texture_id {
            unregister_wgpu_texture_from_egui(frame, texture_id);
        }
    });

    // 画像デコード
    let pixels;
    let width;
    let height;
    meas!({
        (pixels, width, height) = match decode_image_to_rgba8(path) {
            Ok(res) => res,
            Err(e) => {
                eprintln!("Failed to decode image: {}", e);
                return None;
            }
        };
    });

    // wgpuデバイス・キュー取得
    let device;
    let queue;
    meas!({
        let render_state = match frame.wgpu_render_state() {
            Some(rs) => rs,
            None => {
                eprintln!("wgpu_render_state not available");
                return None;
            }
        };
        device = &render_state.device;
        queue = &render_state.queue;
    });

    // テクスチャ作成
    let _texture;
    let view;
    let sampler;
    meas!({
        (_texture, view, sampler) =
            create_gpu_texture_from_rgba8_aligned(device, queue, &pixels, width, height);
    });

    // eguiにテクスチャ登録
    let texture_id;
    meas!({
        texture_id = register_wgpu_texture_to_egui(frame, &view, &sampler);
    });

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
            // if !self.image_files.is_empty() && self.current_index != self.prev_index {
            //     meas!({
            //         if let Some(path) = self.image_files.get(self.current_index) {
            //             let result =
            //                 load_and_register_texture(frame, path, self.current_texture_id.take());
            //             if let Some((texture_id, size)) = result {
            //                 self.current_texture_id = Some(texture_id);
            //                 self.current_size = Some(size);
            //             } else {
            //                 self.current_texture_id = None;
            //                 self.current_size = None;
            //             }
            //         }
            //         self.prev_index = self.current_index;
            //     })
            // }

            // 画像表示
            if let (Some(texture_id), Some((width, height))) =
                (self.current_texture_id, self.current_size)
            {
                // ui.image((texture_id, egui::Vec2::new(width as f32, height as f32)));
                let available = ui.available_size();
                ui.image((texture_id, available));
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
