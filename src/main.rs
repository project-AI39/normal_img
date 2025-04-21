// egui/eframe 関連
use eframe::{egui, wgpu};
use egui::{ColorImage, Context, TextureHandle, TextureId, Ui};

// 画像処理
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageError};

// ファイルパスやファイル操作
use std::fs;
use std::path::PathBuf;

// wgpu 関連
use wgpu::{
    AddressMode, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, Extent3d,
    FilterMode, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, Origin3d, Queue,
    SamplerDescriptor, Texture, TextureAspect, TextureDescriptor, TextureFormat, TextureUsages,
    TextureViewDescriptor,
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
                    if ext == "jpg"
                        || ext == "jpeg"
                        || ext == "png"
                        || ext == "webp"
                        || ext == "gif"
                    {
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
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}
