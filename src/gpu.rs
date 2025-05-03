use eframe::wgpu;
use std::sync::Arc;
use wgpu::{Device, Queue};
/// RGBA8画像データをGPUに転送しテクスチャを作成する（パディング対応）
pub fn create_gpu_texture_from_rgba8_aligned(
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

    let mut padded_data = vec![0u8; padded_bytes_per_row * height as usize];
    for y in 0..height as usize {
        let src_start = y * unpadded_bytes_per_row;
        let src_end = src_start + unpadded_bytes_per_row;
        let dst_start = y * padded_bytes_per_row;
        let dst_end = dst_start + unpadded_bytes_per_row;

        assert!(src_end <= rgba_pixels.len(), "src_end out of bounds");
        assert!(dst_end <= padded_data.len(), "dst_end out of bounds");

        padded_data[dst_start..dst_end].copy_from_slice(&rgba_pixels[src_start..src_end]);
    }

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

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

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
pub fn register_wgpu_texture_to_egui(
    frame: &mut eframe::Frame,
    texture_view: &wgpu::TextureView,
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
// unregister_wgpu_textureの実装
pub fn unregister_wgpu_texture(frame: &mut eframe::Frame, tex_id: egui::TextureId) {
    if let Some(render_state) = frame.wgpu_render_state() {
        let mut renderer = render_state.renderer.write();
        renderer.free_texture(&tex_id);
    }
}
