use eframe::egui::{self, TextureId};
use num_cpus;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use std::sync::mpsc::{self, Receiver, Sender};
use std::{collections::HashMap, path::PathBuf, sync::Arc};

use crate::gpu::{
    create_gpu_texture_from_rgba8_aligned, register_wgpu_texture_to_egui, unregister_wgpu_texture,
};
use crate::image::decode_image_to_rgba8;
use crate::ui::{folder_dialog, navigation_buttons};

struct GpuImageRequest {
    pub path: PathBuf,
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct MyApp {
    pub folder: Option<PathBuf>,
    pub image_files: Vec<PathBuf>,
    pub current_index: usize,
    pub previous_index: usize,
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
                    let (_tex, view, _sampler) = create_gpu_texture_from_rgba8_aligned(
                        std::sync::Arc::new(device),
                        std::sync::Arc::new(queue),
                        &req.pixels,
                        req.width,
                        req.height,
                    );
                    // register_wgpu_texture_to_eguiを使ってTextureIdを取得
                    let tex_id = register_wgpu_texture_to_egui(frame, &view);
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
            navigation_buttons(self, ui, false);
            if !self.image_files.is_empty() && self.current_index != self.previous_index {
                on_image_index_changed(self, frame);
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

fn on_image_index_changed(app: &mut MyApp, frame: &mut eframe::Frame) {
    if app.image_files.is_empty() {
        return;
    }
    build_prefetch_list(app);
    release_unused_textures(app, frame);
    println!("Prefetching images: {:?}", app.priority_list);
    prefetch_tasks_multipool(app);
}

fn release_unused_textures(app: &mut MyApp, frame: &mut eframe::Frame) {
    // プリフェッチ範囲のパスをセット化
    let priority_set: std::collections::HashSet<_> = app.priority_list.iter().cloned().collect();
    // 削除対象を収集
    let unused: Vec<_> = app
        .texture_cache
        .keys()
        .filter(|path| !priority_set.contains(*path))
        .cloned()
        .collect();
    // テクスチャ解放
    for path in unused {
        if let Some(tex_id) = app.texture_cache.remove(&path) {
            unregister_wgpu_texture(frame, tex_id);
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
