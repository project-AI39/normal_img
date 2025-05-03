use rfd::FileDialog;
use std::path::{Path, PathBuf};

use crate::app::MyApp;
use egui::Ui;

/// フォルダ選択ダイアログを表示し、画像ファイル一覧を取得
pub fn folder_dialog(app: &mut MyApp, ui: &mut Ui) {
    if ui.button("Open Folder").clicked() {
        if let Some(folder) = FileDialog::new().pick_folder() {
            app.folder = Some(folder.clone());
            app.image_files = get_image_files(&folder);
            app.current_index = 0;
            app.previous_index = usize::MAX;
        }
    }
}

/// 指定ディレクトリ内の画像ファイル一覧を取得
fn get_image_files(dir: &Path) -> Vec<PathBuf> {
    const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp"];
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.is_file()
                        && path
                            .extension()
                            .and_then(|ext| {
                                let ext = ext.to_string_lossy().to_lowercase();
                                IMAGE_EXTENSIONS.contains(&ext.as_str()).then_some(())
                            })
                            .is_some()
                })
                .collect()
        })
        .unwrap_or_default()
}

/// 画像の前後移動ボタン
pub fn navigation_buttons(app: &mut MyApp, ui: &mut Ui, loop_enabled: bool) {
    if app.image_files.is_empty() {
        return;
    }
    ui.horizontal(|ui| {
        if ui.button("Previous").clicked() {
            if app.current_index > 0 {
                app.current_index -= 1;
            } else if loop_enabled {
                app.current_index = app.image_files.len() - 1;
            }
        }
        if ui.button("Next").clicked() {
            if app.current_index + 1 < app.image_files.len() {
                app.current_index += 1;
            } else if loop_enabled {
                app.current_index = 0;
            }
        }
    });
}
