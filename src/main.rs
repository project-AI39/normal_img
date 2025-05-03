mod app;
mod gpu;
mod image;
mod ui;

use app::MyApp;
use eframe;

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
