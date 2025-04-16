use eframe::{self, egui};
use rfd::FileDialog;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "画像ビュアー",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Box::new(ImageViewer::default())
        }),
    )
}

#[derive(Default)]
struct ImageViewer {
    directory_path: Option<String>,
    images: Vec<PathBuf>,
    current_image_index: usize,
}

impl ImageViewer {
    fn load_images_from_directory(&mut self, path: PathBuf) {
        let dir = match fs::read_dir(&path) {
            Ok(dir) => dir,
            Err(_) => return,
        };

        self.images.clear();

        for item in dir {
            if let Ok(entry) = item {
                let path = entry.path();
                if path.is_file() {
                    if let Some(extension) = path.extension() {
                        // 画像ファイルのみを追加
                        if let Some(ext) = extension.to_str() {
                            let ext = ext.to_lowercase();
                            if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "gif" {
                                self.images.push(path);
                            }
                        }
                    }
                }
            }
        }

        self.images.sort();
        self.current_image_index = if self.images.is_empty() { 0 } else { 0 };
    }

    fn next_image(&mut self) {
        if self.images.is_empty() {
            return;
        }

        if self.current_image_index < self.images.len() - 1 {
            self.current_image_index += 1;
        } else {
            self.current_image_index = 0;
        }
    }

    fn prev_image(&mut self) {
        if self.images.is_empty() {
            return;
        }

        if self.current_image_index > 0 {
            self.current_image_index -= 1;
        } else {
            self.current_image_index = self.images.len() - 1;
        }
    }
}

impl eframe::App for ImageViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // トップバー
            ui.horizontal(|ui| {
                if ui.button("フォルダを開く").clicked() {
                    if let Some(path) = FileDialog::new().pick_folder() {
                        self.directory_path = Some(path.display().to_string());
                        self.load_images_from_directory(path);
                    }
                }

                let has_images = !self.images.is_empty();
                ui.add_enabled_ui(has_images, |ui| {
                    if ui.button("←").clicked() {
                        self.prev_image();
                    }

                    if ui.button("→").clicked() {
                        self.next_image();
                    }
                });

                // 現在のディレクトリを表示
                if let Some(dir) = &self.directory_path {
                    ui.label(format!("ディレクトリ: {}", dir));
                }
            });

            // 画像表示領域
            ui.add_space(10.0);

            let central_panel_height = ui.available_height();
            let central_panel_width = ui.available_width();

            if !self.images.is_empty() {
                let image_path = &self.images[self.current_image_index];
                if let Some(path_str) = image_path.to_str() {
                    let image_url = format!("file://{}", path_str);

                    // 画像を中央に表示し、UIに合わせてサイズ調整
                    ui.vertical_centered(|ui| {
                        ui.add(egui::Image::new(image_url).fit_to_exact_size(egui::vec2(
                            central_panel_width * 0.9,
                            central_panel_height * 0.9,
                        )));
                    });

                    // 画像情報表示
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "画像 {}/{}：{}",
                            self.current_image_index + 1,
                            self.images.len(),
                            image_path
                                .file_name()
                                .and_then(|f| f.to_str())
                                .unwrap_or("不明なファイル名")
                        ));
                    });
                }
            } else {
                // 画像がない場合
                ui.vertical_centered(|ui| {
                    ui.add_space(central_panel_height * 0.4);
                    ui.label("画像が選択されていません。フォルダを開いてください。");
                });
            }
        });
    }
}
