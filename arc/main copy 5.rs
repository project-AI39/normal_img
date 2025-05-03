// --- 標準ライブラリ ---
use std::sync::mpsc::{Receiver, Sender, channel};
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
// --- 外部クレート ---
use eframe::{
    NativeOptions,
    egui::{self, TextureId},
    wgpu::{
        self, AddressMode, Device, Extent3d, FilterMode, Origin3d, Queue, Sampler,
        SamplerDescriptor, Texture, TextureAspect, TextureDescriptor, TextureDimension,
        TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    },
};
use image::{ImageError, flat::View};
use libwebp::{WebPDecodeRGBA, WebPGetInfo, error::WebPSimpleError as WebPError};
use multipool::ThreadPoolBuilder;
use multipool::pool::ThreadPool;
use multipool::pool::modes::PriorityGlobalQueueMode;
use num_cpus;
use png::{BitDepth, ColorType, Decoder, Encoder};
use rfd::FileDialog;
use turbojpeg::{Decompressor, Image, PixelFormat, Subsamp, compress_image, decompress_image};

#[derive(Debug)]
struct PrefetchTask {
    index: usize,
    priority: usize, // 近いほど小さい
    is_cached: bool,
}
// BinaryHeapでpriority小さい順にするための比較実装
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

// --- プリフェッチタスクハンドル ---
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

// --- 画像デコードエラー ---
#[derive(Debug)]
pub enum ImageDecodeError {
    Io(std::io::Error),
    Png(png::DecodingError),
    Jpeg(turbojpeg::Error),
    WebP(libwebp::error::WebPSimpleError),
    UnsupportedFormat,
}

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

// 画像リスト管理
struct ImageList {
    files: Vec<PathBuf>,
    current_index: usize,
    prev_index: usize,
}

impl ImageList {
    fn new() -> Self {
        Self {
            files: Vec::new(),
            current_index: 0,
            prev_index: usize::MAX,
        }
    }
    fn get_current_file(&self) -> Option<&PathBuf> {
        self.files.get(self.current_index)
    }
    fn set_files(&mut self, files: Vec<PathBuf>) {
        self.files = files;
        self.current_index = 0;
        self.prev_index = usize::MAX;
    }
    fn next(&mut self) {
        if self.current_index + 1 < self.files.len() {
            self.current_index += 1;
        }
    }
    fn prev(&mut self) {
        if self.current_index > 0 {
            self.current_index -= 1;
        }
    }
}

// VRAMキャッシュ管理
struct VramCacheEntry {
    texture_id: egui::TextureId,
    size_bytes: usize,
    width: u32,
    height: u32,
}

struct VramCacheManager {
    cache: HashMap<usize, VramCacheEntry>,
    total_bytes: usize,
    limit_bytes: usize,
}
struct TextureRegisterRequest {
    index: usize,
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    width: u32,
    height: u32,
}

impl VramCacheManager {
    fn new(limit_bytes: usize) -> Self {
        Self {
            cache: HashMap::new(),
            total_bytes: 0,
            limit_bytes,
        }
    }
    fn check_cache(&mut self) {
        self.total_bytes = self.cache.values().map(|e| e.size_bytes).sum();
    }
    fn release_unused(&mut self, prefetch_list: &[PrefetchTask], frame: &mut eframe::Frame) {
        let prefetch_indices: HashSet<_> = prefetch_list.iter().map(|t| t.index).collect();
        let to_remove: Vec<_> = self
            .cache
            .keys()
            .filter(|idx| !prefetch_indices.contains(idx))
            .cloned()
            .collect();
        for idx in to_remove {
            if let Some(entry) = self.cache.remove(&idx) {
                unregister_wgpu_texture_from_egui(frame, entry.texture_id);
            }
        }
    }
}
pub struct MyApp {
    selected_folder: Option<String>,
    image_list: ImageList,
    vram_manager: VramCacheManager,
    current_texture_id: Option<egui::TextureId>,
    current_size: Option<(u32, u32)>,
    prefetch_tasks: Vec<PrefetchTaskHandle>,
    prefetch_queue: Arc<Mutex<BinaryHeap<PrefetchTask>>>,
    texture_register_tx: Sender<TextureRegisterRequest>,
    texture_register_rx: Receiver<TextureRegisterRequest>,
    pool: ThreadPool<PriorityGlobalQueueMode>,
}

impl Default for MyApp {
    fn default() -> Self {
        let (tx, rx) = channel();
        Self {
            selected_folder: None,
            image_list: ImageList::new(),
            vram_manager: VramCacheManager::new(512 * 1024 * 1024), // 512MB
            current_texture_id: None,
            current_size: None,
            prefetch_tasks: Vec::new(),
            prefetch_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            texture_register_tx: tx,
            texture_register_rx: rx,
            pool: create_prefetch_pool(),
        }
    }
}

/// 指定ディレクトリから対応画像ファイル(PNG/JPEG/WebP)の一覧を取得する
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
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    buf.truncate(info.buffer_size());
    Ok((buf, info.width, info.height))
}

/// JPEG画像のデコード
fn decode_jpeg_to_rgba8(path: &Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
    let data = std::fs::read(path)?;
    let mut decompressor = turbojpeg::Decompressor::new()?;
    let header = decompressor.read_header(&data)?;

    let mut image = turbojpeg::Image {
        pixels: vec![0; (header.width * header.height * 4) as usize],
        width: header.width,
        pitch: header.width * 4,
        height: header.height,
        format: turbojpeg::PixelFormat::RGBA,
    };

    decompressor.decompress(&data, image.as_deref_mut())?;
    Ok((image.pixels, image.width as u32, image.height as u32))
}

/// WebP画像のデコード
fn decode_webp_to_rgba8(path: &Path) -> Result<(Vec<u8>, u32, u32), ImageDecodeError> {
    let data = std::fs::read(path)?;
    let (width, height, buf) = libwebp::WebPDecodeRGBA(&data)?;
    Ok((buf.to_vec(), width, height))
}

/// 画像ファイルの幅・高さからVRAM使用量（パディング考慮）を計算
fn get_image_size_bytes(path: &Path) -> Option<usize> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    let (width, height) = match ext.as_str() {
        "png" => {
            let file = File::open(path).ok()?;
            let mut reader = BufReader::new(file);
            let mut decoder = png::Decoder::new(&mut reader);
            let info = decoder.read_header_info().ok()?;
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
            let (width, height) = WebPGetInfo(&data).unwrap();
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

/// VRAMキャッシュの総使用量を再計算する
fn vram_cache_check(app: &mut MyApp) {
    app.vram_manager.total_bytes = app.vram_manager.cache.values().map(|e| e.size_bytes).sum();
}

/// プリフェッチ対象外となったタスクをキャンセル・削除する
fn cleanup_old_prefetch_tasks(prefetch_list: &[PrefetchTask], app: &mut MyApp) {
    let prefetch_indices: HashSet<_> = prefetch_list.iter().map(|t| t.index).collect();
    for task in app.prefetch_tasks.iter_mut() {
        if !prefetch_indices.contains(&task.index) {
            task.cancel();
        }
    }
    app.prefetch_tasks.retain(|task| !task.is_cancelled());
}

// multipoolのスレッドプールを初期化
fn create_prefetch_pool() -> ThreadPool<PriorityGlobalQueueMode> {
    ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .enable_priority()
        .build()
}
fn start_prefetch_tasks_multipool(
    prefetch_list: &[PrefetchTask],
    app: &mut MyApp,
    frame: &mut eframe::Frame,
) {
    // 必要なwgpuデバイス・キューを事前に取得
    let render_state = match frame.wgpu_render_state() {
        Some(rs) => rs.clone(),
        None => {
            eprintln!("wgpu_render_state not available");
            return;
        }
    };
    let device = Arc::new(render_state.device.clone());
    let queue = Arc::new(render_state.queue.clone());

    for task in prefetch_list {
        if task.is_cached {
            continue;
        }
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let image_files = app.image_list.files.clone();
        let idx = task.index;
        let priority = task.priority;
        let tx = app.texture_register_tx.clone();
        let device = device.clone();
        let queue = queue.clone();
        let cancel_flag_clone = cancel_flag.clone();

        // ここでapp.poolを直接参照
        app.pool.spawn_with_priority(
            move || {
                if cancel_flag_clone.load(Ordering::SeqCst) {
                    return;
                }
                let (pixels, width, height) = match decode_image_to_rgba8(&image_files[idx]) {
                    Ok(res) => res,
                    Err(_) => return,
                };
                if cancel_flag_clone.load(Ordering::SeqCst) {
                    return;
                }
                let (_texture, view, sampler) = create_gpu_texture_from_rgba8_aligned(
                    device.clone(),
                    queue.clone(),
                    &pixels,
                    width,
                    height,
                );
                let _ = tx.send(TextureRegisterRequest {
                    index: idx,
                    texture_view: view,
                    sampler,
                    width,
                    height,
                });
            },
            priority,
        );

        app.prefetch_tasks.push(PrefetchTaskHandle {
            index: idx,
            cancel_flag,
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

    // GPUへ画像データ転送
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

    // テクスチャビュー作成
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // サンプラー作成
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

/// wgpuテクスチャビューとサンプラーをeguiに登録し、TextureIdを返す
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

/// eguiからテクスチャを解放する
fn unregister_wgpu_texture_from_egui(frame: &mut eframe::Frame, texture_id: egui::TextureId) {
    if let Some(render_state) = frame.wgpu_render_state() {
        let mut renderer = render_state.renderer.write();
        renderer.free_texture(&texture_id);
    }
}

fn build_prefetch_list(
    image_list: &ImageList,
    vram_manager: &VramCacheManager,
) -> Vec<PrefetchTask> {
    let mut list = Vec::new();
    let mut used_bytes = vram_manager.total_bytes;
    let vram_limit = vram_manager.limit_bytes;
    let mut indices: Vec<usize> = (0..image_list.files.len()).collect();
    indices.sort_by_key(|&i| (i as isize - image_list.current_index as isize).abs());
    for &idx in &indices {
        if vram_manager.cache.contains_key(&idx) {
            list.push(PrefetchTask {
                index: idx,
                priority: (idx as isize - image_list.current_index as isize).abs() as usize,
                is_cached: true,
            });
            continue;
        }
        let size_bytes = match get_image_size_bytes(&image_list.files[idx]) {
            Some(sz) => sz,
            None => continue,
        };
        if used_bytes + size_bytes > vram_limit {
            break;
        }
        used_bytes += size_bytes;
        list.push(PrefetchTask {
            index: idx,
            priority: (idx as isize - image_list.current_index as isize).abs() as usize,
            is_cached: false,
        });
    }
    list
}

fn release_unused_vram_images(
    prefetch_list: &[PrefetchTask],
    vram_manager: &mut VramCacheManager,
    frame: &mut eframe::Frame,
) {
    vram_manager.release_unused(prefetch_list, frame);
}

// --- 画像インデックス変更時にキャッシュ・プリフェッチを制御
fn on_image_index_changed(app: &mut MyApp, frame: &mut eframe::Frame) {
    app.vram_manager.check_cache();
    let prefetch_list = build_prefetch_list(&app.image_list, &app.vram_manager);
    release_unused_vram_images(&prefetch_list, &mut app.vram_manager, frame);
    cleanup_old_prefetch_tasks(&prefetch_list, app);
    start_prefetch_tasks_multipool(&prefetch_list, app, frame);
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // バックグラウンドスレッドからの登録要求を受信して処理
        while let Ok(req) = self.texture_register_rx.try_recv() {
            let texture_id = register_wgpu_texture_to_egui(frame, &req.texture_view, &req.sampler);
            let bytes_per_pixel = 4;
            let unpadded_bytes_per_row = req.width as usize * bytes_per_pixel;
            let align = 256;
            let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
            let size_bytes = padded_bytes_per_row * req.height as usize;
            self.vram_manager.cache.insert(
                req.index,
                VramCacheEntry {
                    texture_id,
                    size_bytes,
                    width: req.width,
                    height: req.height,
                },
            );
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // フォルダ選択
            if ui.button("Select Folder").clicked() {
                if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                    let folder_str = folder.display().to_string();
                    self.selected_folder = Some(folder_str.clone());
                    self.image_list.set_files(get_image_files(&folder_str));
                }
            }

            // 画像送りUI
            if !self.image_list.files.is_empty() {
                ui.horizontal(|ui| {
                    if ui.button("Previous").clicked() {
                        self.image_list.prev();
                    }
                    ui.label(format!(
                        "Image {}/{}",
                        self.image_list.current_index + 1,
                        self.image_list.files.len()
                    ));
                    if ui.button("Next").clicked() {
                        self.image_list.next();
                    }
                });
            }

            // 画像切り替え時のみロード＆登録
            if !self.image_list.files.is_empty()
                && self.image_list.current_index != self.image_list.prev_index
            {
                on_image_index_changed(self, frame);
                self.image_list.prev_index = self.image_list.current_index;
            }

            // 画像表示
            if let (Some(texture_id), Some((width, height))) =
                (self.current_texture_id, self.current_size)
            {
                let available = ui.available_size();
                let scale = (available.x / width as f32)
                    .min(available.y / height as f32)
                    .min(1.0);
                let size = egui::Vec2::new(width as f32 * scale, height as f32 * scale);
                ui.image((texture_id, size));
            }
        });
    }
}
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
