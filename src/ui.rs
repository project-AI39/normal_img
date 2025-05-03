use rfd::FileDialog;
use std::path::{Path, PathBuf};

use crate::app::MyApp;
use egui::{Color32, Frame, Mesh, Pos2, Rect, Response, TextureId, Ui, Vec2};

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
pub fn show_image_viewer_mesh_interactive(
    ui: &mut egui::Ui,
    tex_id: egui::TextureId,
    zoom: &mut f32,
    pan: &mut egui::Vec2,
    rotation_rad: &mut f32,
    image_width: f32,
    image_height: f32,
) {
    let available_size = ui.available_size();

    // アスペクト比を維持した最大サイズを計算
    let image_aspect = image_width / image_height;
    let area_aspect = available_size.x / available_size.y;
    let fit_size = if area_aspect > image_aspect {
        // エリアが横長→高さ基準
        egui::Vec2::new(available_size.y * image_aspect, available_size.y)
    } else {
        // エリアが縦長→幅基準
        egui::Vec2::new(available_size.x, available_size.x / image_aspect)
    };
    let fit_size = fit_size * *zoom;

    Frame::NONE
        .outer_margin(egui::Margin::ZERO)
        .inner_margin(egui::Margin::ZERO)
        .stroke(egui::Stroke::NONE)
        .fill(Color32::from_rgb(60, 60, 60))
        .show(ui, |ui| {
            let (rect, response) = ui.allocate_exact_size(available_size, egui::Sense::drag());
            let center = rect.center() + *pan;

            // --- ユーザー操作 ---
            if response.hovered() {
                let scroll = ui.input(|i| i.raw_scroll_delta.y);
                if scroll != 0.0 {
                    let prev_zoom = *zoom;
                    let new_zoom = (*zoom * (1.0 + scroll * 0.002)).clamp(0.05, 20.0);

                    if let Some(mouse_pos) = ui.ctx().pointer_hover_pos() {
                        let image_center = rect.center() + *pan;
                        let mouse_to_image_center = mouse_pos - image_center;
                        *pan += mouse_to_image_center * (1.0 - new_zoom / prev_zoom);
                    }

                    *zoom = new_zoom;
                }
            }
            if response.dragged_by(egui::PointerButton::Primary) {
                *pan += response.drag_delta();
            }
            if response.dragged_by(egui::PointerButton::Secondary) {
                *rotation_rad += response.drag_delta().x * 0.005;
            }

            // 画像の半サイズ（アスペクト比維持）
            let half_size = 0.5 * fit_size;

            // 回転行列
            let rot = |v: egui::Vec2| {
                let (sin, cos) = rotation_rad.sin_cos();
                egui::Vec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos)
            };

            // 四隅のローカル座標
            let corners = [
                egui::Vec2::new(-half_size.x, -half_size.y),
                egui::Vec2::new(half_size.x, -half_size.y),
                egui::Vec2::new(half_size.x, half_size.y),
                egui::Vec2::new(-half_size.x, half_size.y),
            ];

            // UV座標
            let uvs = [
                egui::Pos2::new(0.0, 0.0),
                egui::Pos2::new(1.0, 0.0),
                egui::Pos2::new(1.0, 1.0),
                egui::Pos2::new(0.0, 1.0),
            ];

            // 頂点を回転・移動
            let points: Vec<egui::Pos2> = corners
                .iter()
                .map(|&v| center + rot(v))
                .map(|v| egui::Pos2::new(v.x, v.y))
                .collect();

            // Mesh作成
            let mut mesh = egui::Mesh::with_texture(tex_id);
            for i in 0..4 {
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: points[i],
                    uv: uvs[i],
                    color: egui::Color32::WHITE,
                });
            }
            mesh.indices.extend([0, 1, 2, 0, 2, 3]);
            ui.painter_at(rect).add(mesh);
        });
}
