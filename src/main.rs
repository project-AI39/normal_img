use eframe::{self, egui};
use rfd::FileDialog;
use std::path::PathBuf;

// 画像描写システムモジュール
mod image_renderer {
    use egui::{Ui, Vec2};
    use std::path::PathBuf;

    // サポートする画像拡張子
    const SUPPORTED_EXTENSIONS: [&str; 4] = ["jpg", "jpeg", "png", "gif"];

    /// 画像表示オプション
    #[derive(Debug, Clone)]
    pub struct RenderOptions {
        pub size: Vec2,            // 表示サイズ
        pub maintain_aspect: bool, // アスペクト比維持
        pub fit_to_screen: bool,   // 画面フィット
    }

    impl Default for RenderOptions {
        fn default() -> Self {
            Self {
                size: Vec2::new(800.0, 600.0), // デフォルトサイズ
                maintain_aspect: true,
                fit_to_screen: true,
            }
        }
    }

    /// 画像ファイルか判定
    pub fn is_image_file(path: &PathBuf) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
            .unwrap_or(false)
    }

    /// パスから画像URL生成
    pub fn path_to_image_url(path: &PathBuf) -> Option<String> {
        path.to_str()
            .map(|path_str| format!("file://{}", path_str.replace('\\', "/")))
    }

    /// 画像表示関数
    pub fn render_image(ui: &mut Ui, image_path: Option<&PathBuf>, options: &RenderOptions) {
        match image_path.and_then(path_to_image_url) {
            Some(image_url) => {
                ui.vertical_centered(|ui| {
                    let mut image = egui::Image::new(image_url);

                    if options.fit_to_screen {
                        if options.maintain_aspect {
                            image = image.fit_to_exact_size(options.size);
                        } else {
                            image = image.max_size(options.size);
                        }
                    }

                    ui.add(image);
                });
            }
            None => {
                ui.vertical_centered(|ui| {
                    ui.add_space(options.size.y * 0.4);
                    ui.label("No images selected. Please open a folder.");
                });
            }
        }
    }

    /// 画像情報表示用構造体
    pub struct ImageInfo<'a> {
        pub current_index: usize,
        pub total_images: usize,
        pub filename: &'a str,
    }

    /// 画像情報表示関数
    pub fn render_image_info(ui: &mut Ui, info: Option<ImageInfo>) {
        if let Some(info) = info {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Image {}/{}: {}",
                    info.current_index + 1,
                    info.total_images,
                    info.filename
                ));
            });
        }
    }
}

// 画像管理モジュール
mod image_manager {
    use super::image_renderer;
    use std::{fs, path::PathBuf};

    /// ディレクトリから画像を読み込み
    pub fn load_images_from_directory(path: &PathBuf) -> Vec<PathBuf> {
        let mut images = Vec::new();

        if let Ok(dir) = fs::read_dir(path) {
            for entry in dir.filter_map(Result::ok) {
                let path = entry.path();
                if path.is_file() && image_renderer::is_image_file(&path) {
                    images.push(path);
                }
            }
            images.sort();
        }

        images
    }
}

/// メインアプリケーション構造体
#[derive(Default)]
struct ImageViewer {
    directory_path: Option<String>,
    images: Vec<PathBuf>,
    current_image_index: usize,
}

impl ImageViewer {
    /// ディレクトリから画像を読み込み
    fn load_images_from_directory(&mut self, path: PathBuf) {
        self.images = image_manager::load_images_from_directory(&path);
        self.directory_path = Some(path.display().to_string());
        self.current_image_index = if self.images.is_empty() { 0 } else { 0 };
    }

    /// 次の画像へ移動
    fn next_image(&mut self) {
        if !self.images.is_empty() {
            self.current_image_index = (self.current_image_index + 1) % self.images.len();
        }
    }

    /// 前の画像へ移動
    fn prev_image(&mut self) {
        if !self.images.is_empty() {
            self.current_image_index = self
                .current_image_index
                .checked_sub(1)
                .unwrap_or(self.images.len() - 1);
        }
    }

    /// 現在の画像パス取得
    fn current_image(&self) -> Option<&PathBuf> {
        self.images.get(self.current_image_index)
    }

    /// 画像情報取得
    fn current_image_info(&self) -> Option<image_renderer::ImageInfo> {
        self.current_image().map(|path| image_renderer::ImageInfo {
            current_index: self.current_image_index,
            total_images: self.images.len(),
            filename: path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("Unknown filename"),
        })
    }

    /// ツールバー表示
    fn display_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // フォルダオープンボタン
            if ui.button("Open Folder").clicked() {
                if let Some(path) = FileDialog::new().pick_folder() {
                    self.load_images_from_directory(path);
                }
            }

            // ナビゲーションボタン
            ui.add_enabled_ui(!self.images.is_empty(), |ui| {
                if ui.button("<").clicked() {
                    self.prev_image();
                }
                if ui.button(">").clicked() {
                    self.next_image();
                }
            });

            // 現在のディレクトリ表示
            if let Some(dir) = &self.directory_path {
                ui.label(format!("Directory: {}", dir));
            }
        });
    }
}

impl eframe::App for ImageViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // ツールバー表示
            self.display_toolbar(ui);
            ui.add_space(10.0);

            // 表示領域サイズ取得
            let available_size = ui.available_size() * 0.9;

            // 画像表示オプション設定
            let render_options = image_renderer::RenderOptions {
                size: available_size,
                ..Default::default()
            };

            // 画像表示
            image_renderer::render_image(ui, self.current_image(), &render_options);

            // 画像情報表示
            image_renderer::render_image_info(ui, self.current_image_info());
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    // ウィンドウ設定
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    // アプリケーション起動
    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(|cc| {
            // 画像ローダー初期化
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(ImageViewer::default()))
        }),
    )
}
