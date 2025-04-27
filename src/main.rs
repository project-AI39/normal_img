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
use multipool::ThreadPoolBuilder;
use multipool::pool::ThreadPool;
use multipool::pool::modes::PriorityGlobalQueueMode;
use num_cpus;
use png::{BitDepth, ColorType, Decoder, Encoder};
use rfd::FileDialog;
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
use turbojpeg::{Decompressor, Image, PixelFormat, Subsamp, compress_image, decompress_image};
pub struct MyApp {
    folder: Option<PathBuf>,
    image_files: Vec<PathBuf>,
    current_index: usize,
    previous_index: usize,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            folder: None,
            image_files: Vec::new(),
            current_index: 0,
            previous_index: 0,
        }
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
impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // バックグラウンドスレッドからの登録要求を受信して処理
        egui::CentralPanel::default().show(ctx, |ui| {
            folder_dialog(self, ui);
            navigation_buttons(self, ui);
            if !self.image_files.is_empty() && self.current_index != self.previous_index {
                on_image_index_changed(self, frame);
            }
        });
    }
}

fn on_image_index_changed(app: &mut MyApp, frame: &mut eframe::Frame) {}

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
