#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    clippy::manual_slice_size_calculation,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    unused_assignments,
    unused_must_use
)]
#![feature(portable_simd)]
//
// Internal Dep:
//

mod lib;  // Declare the lib module
use lib::SV64tree;
/* Unused brickmaps!
use lib::brickmap;  // Use the Brickmap struct
use lib::brickmap::ChunkWorld;
use lib::brickmap::GPUChunk;
*/
//
// Dependancies
//

use bytemuck::Zeroable;
use bytemuck::Pod;
//dot vox stuff:
use dot_vox::DotVoxData;
use dot_vox::load;
//textures
mod mc;
mod texture;

//math
use cgmath::num_traits::int;
use cgmath::Vector3;
use cgmath::vec2;
use cgmath::vec3;
use cgmath::Vector2;
use cgmath::{InnerSpace, Deg};

use dot_vox::Size;
use geese::Mut;
use geese::SystemRef;
//geese
use geese::{
    dependencies, Dependencies, EventHandlers, EventQueue,
    GeeseContext, GeeseContextHandle, GeeseSystem, event_handlers,
};

use lib::SV64tree::add_vox_to_tree;
use lib::SV64tree::create_test_tree;
use lib::SV64tree::create_test_tree_from_vox;
use lib::SV64tree::Sparse64Tree;
use lib::SV64tree::Tree64GpuManager;
use lib::SV64tree::TreeMemoryManager;
use lib::SV64tree::AABB;
use noise::Perlin;
use wgpu::core::device;
use winit::keyboard::Key;
//use wgpu::hal::vulkan::Buffer;
//use texture::Texture;
//use wgpu::core::device;
use std::collections::btree_map::Range;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::{default, iter, usize};
//wgpu
use dot_vox::Color;
use wgpu::util::DeviceExt;
use wgpu::{Adapter, BindGroup, BindGroupLayout, Buffer, Device, Instance, PipelineCompilationOptions, Queue, RenderPipeline, Surface, SurfaceConfiguration, Texture, TextureView};
use wgpu::core::device::queue;
//winit
use winit::dpi::PhysicalSize;
use winit::{ event_loop, event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\

//
// Startup + Eventloop
//
//static vox_path: &str = "blue_shroom_hut.vox";
static vox_path: &str = "C:\\Users\\USER\\.vox";
//main loop
fn main() {
    println!("Size of CameraBuffer: {} bytes", std::mem::size_of::<CameraBuffer>());
    println!("Alignment of CameraBuffer: {} bytes", std::mem::align_of::<CameraBuffer>());
    println!("Program started.");
    pollster::block_on(run());
    println!("Program ended.");
}

mod on {
    use std::fmt::DebugTuple;

    use winit::dpi::PhysicalSize;

    pub struct NewFrame{
    }

    pub struct WindowResized{
        pub physical_size: PhysicalSize<u32>,
    }

    pub struct MouseMoved{
        pub delta_y: f32,
        pub delta_x: f32,
    }

    pub struct KeyPressed {
        pub key: winit::keyboard::KeyCode,
        pub state: winit::event::ElementState,
    }

}

async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new()
        .with_maximized(true)
        .with_title("Lepton Engine")
        .build(&event_loop)
        .unwrap());
    window.set_cursor_visible(false);
    window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
        .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
        .expect("Failed to grab cursor");

    let ctx = GeeseContext::default();
    let ctx = Arc::new(Mutex::new(ctx));
    
    {
        let mut ctx_guard = ctx.lock().unwrap();

        ctx_guard.flush().with(geese::notify::add_system::<ParamSystem>());
        ctx_guard.get_mut::<ParamSystem>().window = Some(window.clone());

        ctx_guard.flush()
            .with(geese::notify::add_system::<InstanceSystem>())
            .with(geese::notify::add_system::<SurfaceSystem>())
            .with(geese::notify::add_system::<DeviceSystem>())
            .with(geese::notify::add_system::<CameraSystem>())
            .with(geese::notify::add_system::<PipelineSystem>())
            .with(geese::notify::add_system::<ResizeSystem>())
            .with(geese::notify::add_system::<RenderSystem>())
            .with(geese::notify::add_system::<ComputePipelineSystem>())
            .with(geese::notify::add_system::<WorldSystem>())
            .with(geese::notify::add_system::<FreeCamSystem>())
            /*.with(geese::notify::add_system::<CameraUpdateSystem>())*/;
            
    }
    
    let render_system = {let ctx_guard = ctx.lock().unwrap(); ctx_guard.get::<RenderSystem>();};
    
    let event_loop_ctx = Arc::clone(&ctx);

    let _ = 
    event_loop.run(move |event, control_flow: &event_loop::EventLoopWindowTarget<()>| {
        match event {

            winit::event::Event::DeviceEvent { event, .. } => {
                if let DeviceEvent::MouseMotion { delta } = event {
                    let dx = delta.0;
                    let dy = delta.1;
                    let delta_x= dx as f32;
                    let delta_y= dy as f32;
                    //println!("Yaw: {}, Pitch: {}", delta.0, delta.1);
                    if let Ok(mut ctx_guard) = event_loop_ctx.lock() {
                        ctx_guard.flush().with(on::MouseMoved{delta_x, delta_y});
                    }
                    window.request_redraw();
                    //camera.update_rotation(delta_x as f32, delta_y as f32);
                    
                    
                    
                }
            }

            // Window-specific events
            winit::event::Event::WindowEvent { event, window_id }
                if window_id == window.id() =>
            {
                match event {
                    // Handle close or escape key to exit
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                ..
                            },
                        ..
                    } => control_flow.exit(),

                    // Handle window resize events
                    WindowEvent::Resized(physical_size) => {
                        log::info!("Window resized: {physical_size:?}");
                        if let Ok(mut ctx_guard) = event_loop_ctx.lock() {
                            ctx_guard.flush().with(on::WindowResized{physical_size});
                        }
                        window.request_redraw();
                        //ctx.raise_event(|sys: &mut ResizeSystem| {
                        //sys.handle_resize(physical_size);
                        //});
                    }

                    // Request a redraw when needed
                    WindowEvent::RedrawRequested => {
                        window.request_redraw();
                        if let Ok(mut ctx_guard) = event_loop_ctx.lock() {
                            //window.request_redraw();
                            ctx_guard.flush().with(on::NewFrame{});
                        }
                        //ctx.raise_event(|sys: &mut RenderSystem| {
                        //    sys.render();
                        //});
                    }

                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(key_code),
                                state,
                                ..
                            },
                        ..
                    } => {
                        // Skip the escape key since it's handled above
                        if key_code != KeyCode::Escape {
                            //window.request_redraw();
                            //println!("yeah it works :D || The key pressed:  {:?}", key_code);
                            if let Ok(mut ctx_guard) = event_loop_ctx.lock() {
                                ctx_guard.flush().with(on::KeyPressed{
                                    key: key_code,
                                    state: state,
                                });
                            }
                        }
                    }

                    
                    _ => {}

                }

                

            } // Handle other events if necessary

            _ => {}
        }
    });
}

//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\

//
// Graphics Init
//



#[derive(Default)]
struct ParamSystem{
    window: Option<Arc<Window>>
}

impl GeeseSystem for ParamSystem {
    fn new(_: GeeseContextHandle<Self>) -> Self {
        Self::default()
    }
}

pub struct InstanceSystem {
    instance: Arc<Instance>,
}
impl GeeseSystem for InstanceSystem {  
    fn new(ctx: geese::GeeseContextHandle<Self>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        //let surface_system = ctx.get::<SurfaceSystem>();
        //let surface = &surface_system.surface;
        
        
        

        Self { instance: Arc::new(instance)}
    }
}
impl InstanceSystem {
    pub fn get(self: &Self) -> Arc<Instance> {
        return  Arc::clone(&self.instance);
    }
}

pub struct SurfaceSystem {
    surface: Surface<'static>,
    adapter: Arc<Adapter>,
}
impl GeeseSystem for SurfaceSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<InstanceSystem>()
        .with::<ParamSystem>();
    fn new(ctx: geese::GeeseContextHandle<Self>) -> Self {
        let instance_system = &ctx.get::<InstanceSystem>();
        let instance = &instance_system.instance;

        let param_system = ctx.get::<ParamSystem>();
        let window = param_system.window.clone().expect("Window not initialized");

        let surface = instance.create_surface(window).unwrap();


        let adapter = pollster::block_on( async{instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap()});
        
        

        Self { surface, adapter: Arc::new(adapter) }
    }
    
}
impl SurfaceSystem {
    pub fn get(self: &Self) -> &wgpu::Surface {
        return &self.surface;
    }
}
pub struct DeviceSystem {
    device: Arc<Device>,
    queue: Arc<Queue>,
}
impl DeviceSystem {
    pub async fn init(
        adapter: Arc<Adapter>,
    ) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        let desired_limits = wgpu::Limits {
            // max_buffer_size: 4 * 1024 * 1024 * 1024, // Request up to 4 GB
            // max_uniform_buffer_binding_size: 64 * 1024, // Request larger uniform buffers THIS USED TO BE 64 * 1024
            // max_storage_buffer_binding_size: 2 * 1024 * 1024 * 1024,
            max_buffer_size: 2 * 1024 * 1024 * 1024, // 4 GB
            max_uniform_buffer_binding_size: 64 * 1024, // 64 KB
            max_storage_buffer_binding_size: 1024 * 1024 * 1024 * 2,
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                    required_limits: desired_limits, //wgpu::Limits::default()
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();
        let limits = adapter.limits();
        println!("Max Buffer Size: {}", limits.max_buffer_size);
        println!("Max Storage Buffer Size: {}", limits.max_storage_buffer_binding_size);
        (Arc::new(device), Arc::new(queue))
    }

    pub async fn get(ctx: GeeseContextHandle<Self>) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        let instance_system = ctx.get::<InstanceSystem>();
        let instance = Arc::clone(&instance_system.instance);

        let surface_system = ctx.get::<SurfaceSystem>();
        let surface = &surface_system.surface;
        let adapter = &surface_system.adapter;

        let (device, queue) =
            pollster::block_on(async { DeviceSystem::init(adapter.clone()).await });
        (device, queue)
    }
}
impl GeeseSystem for DeviceSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<InstanceSystem>().with::<SurfaceSystem>();
    fn new(ctx: geese::GeeseContextHandle<Self>) -> Self {
        let (device, queue) = pollster::block_on(DeviceSystem::get(ctx));
        Self { device, queue }
    }
}
struct TextureViewSystem {
    texture_view: TextureView,
    texture: Texture,
}
impl GeeseSystem for TextureViewSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<DeviceSystem>()
        .with::<ParamSystem>();
        
     fn new(ctx: GeeseContextHandle<Self>) -> Self {
        let param_system = ctx.get::<ParamSystem>();
        let window = param_system.window.clone().expect("Window not initialized");
        let size = window.inner_size();
        let device = &ctx.get::<DeviceSystem>().device;

        let texture_size = wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        };
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("Raytracing Output Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: Default::default(),
        };
        let texture = device.create_texture(&texture_desc);
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self { texture_view, texture}
     }

 }

//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\

//
// Pipelines
//

struct ComputePipelineSystem{
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
}
impl GeeseSystem for ComputePipelineSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<DeviceSystem>()
        .with::<SurfaceSystem>()
        .with::<InstanceSystem>()
        .with::<CameraSystem>()
        .with::<ParamSystem>()
        .with::<TextureViewSystem>()
        .with::<WorldSystem>();

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        // Device, Surface, Adapter
        let device_system = ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let queue = &device_system.queue;
        let surface_system = ctx.get::<SurfaceSystem>();
        let surface = &surface_system.surface;
        let adapter = &surface_system.adapter;
        let param_system = ctx.get::<ParamSystem>();
        //Window
        let window = param_system.window.clone().expect("Window not initialized");
        let size = window.inner_size();
        let texture_view = &ctx.get::<TextureViewSystem>().texture_view;
        // Camera
        let camera_system = ctx.get::<CameraSystem>();
        let camera_bind_group_layout = &camera_system.camera_bind_group_layout;
        // // Voxel Data
        
        let world_system = ctx.get::<WorldSystem>();
        let world_bind_group = &world_system.tree_manager.contree_bind_group;
        let world_bind_group_layout = &world_system.tree_manager.contree_bind_group_layout;
        // let gpu_chunk = &chunk_system.gpu_voxel_world;
        // let voxel_world_bind_group_layout   = &gpu_chunk.vox_world_bind_group_layout;
        // let (gpu_bricks, gpu_indices) = chunk_world.collect_bricks_gpu(256);
        // let (bricks_gpu_buffer, indices_gpu_buffer) = upload_to_gpu(&device, &queue, gpu_bricks, gpu_indices);
        





        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/compute_trace.wgsl"));
        
        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);


        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            desired_maximum_frame_latency: 1,
            view_formats: vec![],
        };


        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            
            /*wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { 
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None
                },
                count: None,
            },*/
            
        ],
    });


        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout, &camera_bind_group_layout, &world_bind_group_layout], // Bind group layout created earlier
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "cs_main",
            cache: None,
            compilation_options: Default::default(),
            
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },/*wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &camera_buffer,
                    offset: 0,
                    size: None
                }),
            },*/],
        });

        


        Self { compute_pipeline, compute_bind_group }
    }
}

//Vertex definition.
//Used in rendering the final image.
//No other use use.

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    uv: [f32; 2],
}

// lib.rs
impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [1.0, 1.0, 0.0],
        color: [0.0, 0.0, 0.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.0],
        color: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        position: [1.0, -1.0, 0.0],
        color: [0.0, 0.0, 1.0],
        uv: [1.0, 0.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.0],
        color: [1.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        position: [1.0, 1.0, 0.0],
        color: [0.0, 1.0, 0.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0, 0.0],
        color: [0.0, 0.0, 1.0],
        uv: [0.0, 1.0],
    },
];

struct PipelineSystem {
    render_pipeline: RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    config: SurfaceConfiguration,
    size: PhysicalSize<u32>,
    fragment_bind_group: BindGroup,
    /*depth_texture: Arc<Texture>,*/
}
impl GeeseSystem for PipelineSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<DeviceSystem>()
        .with::<SurfaceSystem>()
        .with::<InstanceSystem>()
        .with::<CameraSystem>()
        .with::<TextureViewSystem>()
        .with::<ParamSystem>();

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
       let device_system = ctx.get::<DeviceSystem>();
       let device = &device_system.device;
       let surface_system = ctx.get::<SurfaceSystem>();
       let surface = &surface_system.surface;
       //let instance_system = ctx.get::<InstanceSystem>();
       //let instance = &instance_system.instance;
       let adapter = &surface_system.adapter;
       let param_system = ctx.get::<ParamSystem>();
       let window = param_system.window.clone().expect("Window not initialized");
       let size = window.inner_size();
       let texture_view = &ctx.get::<TextureViewSystem>().texture_view;
       //let camera_system = ctx.get::<CameraSystem>();
       

       let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
        });


    
        
       




       let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/fragment_render.wgsl"));

       let surface_caps = surface.get_capabilities(&adapter);

       let surface_format = surface_caps
       .formats
       .iter()
       .copied()
       .find(|f| f.is_srgb())
       .unwrap_or(surface_caps.formats[0]);

        
    
        //|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
        //|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
        //|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\

        //
        // Config
        //

        let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        desired_maximum_frame_latency: 0,
        view_formats: vec![],
    };
    //let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");
    /*let camera_bind_group_layout =
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
        label: Some("camera_bind_group_layout"),
    });*/


    let fragment_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Fragment Bind Group Layout"),
        entries: &[
            // Texture sampler
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // Texture view
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });


    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: Default::default(),
        lod_max_clamp: Default::default(),
        compare: Default::default(),
        anisotropy_clamp: 1,
        border_color: Default::default(),
        
    });
    
    let fragment_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Fragment Bind Group"),
        layout: &fragment_bind_group_layout,
        entries: &[
            // Bind the sampler
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            // Bind the texture view
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
        ],
    });
    

       let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&fragment_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
            // Useful for optimizing shader compilation on Android
            cache: None,
        });
    
        surface.configure(&device, &config);

        Self {render_pipeline, vertex_buffer, config, size, fragment_bind_group/*depth_texture: Arc::new(depth_texture)*/}
    }
}

//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\

//
// Rendering
//

struct RenderSystem {
    ctx: GeeseContextHandle<Self>,
}

impl GeeseSystem for RenderSystem {
    const DEPENDENCIES: Dependencies = dependencies()
        .with::<DeviceSystem>()
        .with::<PipelineSystem>()
        .with::<CameraSystem>()
        .with::<SurfaceSystem>()
        .with::<ComputePipelineSystem>()
        .with::<WorldSystem>()
        .with::<ParamSystem>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers().with(Self::call_render);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self {ctx}
    }
}

impl RenderSystem {

    fn call_render(&mut self, event: &on::NewFrame){
        self.render();
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> 
    {
        let device_system = self.ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let queue = &device_system.queue;
        let pipeline_system = self.ctx.get::<PipelineSystem>();
        //let camera_system = self.ctx.get::<CameraSystem>();
        /*let depth_texture = Arc::clone(&pipeline_system.depth_texture);*/
        let surface_system = self.ctx.get::<SurfaceSystem>();
        let surface = &surface_system.surface;

        let output = surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let compute_system = &self.ctx.get::<ComputePipelineSystem>();
        let compute_pipeline = &compute_system.compute_pipeline;
        let compute_bind_group = &compute_system.compute_bind_group;

        let camera_system = self.ctx.get::<CameraSystem>();
        let camera_bind_group = &camera_system.camera_bind_group;
        let camera_bind_group_layout = &camera_system.camera_bind_group_layout;

        let world_system = self.ctx.get::<WorldSystem>();
        let world_bind_group = &world_system.tree_manager.contree_bind_group;
        let world_bind_group_layout = &world_system.tree_manager.contree_bind_group_layout;


        // let chunk_system = self.ctx.get::<ChunkSystem>();
        // let voxel_world = &chunk_system.gpu_voxel_world;
        // let voxel_world_bind_group = &voxel_world.vox_world_bind_group;
        // let voxel_world_bind_group_layout = &voxel_world.vox_world_bind_group_layout;
        

        let param_system = &self.ctx.get::<ParamSystem>();
        let window = param_system.window.clone().expect("Window not initialized");
        let size = window.inner_size();


        let fragment_bind_group = &pipeline_system.fragment_bind_group;

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        
        
    
        // let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("Destination Camera Buffer"),
        //     size: std::mem::size_of::<CameraBuffer>() as u64,
        //     usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        //     mapped_at_creation: false,
        // });
        // //let camera_buffer_array = [camera_buffer];
        // //let src_buffer_data = bytemuck::cast_slice(&camera_buffer_array);
        // let src_buffer_data = bytemuck::bytes_of(&camera_buffer);
        // queue.write_buffer(&src_buffer, 0, &src_buffer_data);
        
        



        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: Default::default(),});
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &compute_bind_group, &[]);
                compute_pass.set_bind_group(1, &camera_bind_group, &[]);
                compute_pass.set_bind_group(2, &world_bind_group, &[]);
                //compute_pass.set_bind_group(2, &voxel_world_bind_group, &[]);
                //compute_pass.set_bind_group(1, &camera_bind_group, &[]);
                compute_pass.dispatch_workgroups((&size.width + 15) / 16, (&size.height + 15) / 16, 1);
        }
        

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&pipeline_system.render_pipeline);
            render_pass.set_bind_group(0, &fragment_bind_group, &[]);
            render_pass.set_vertex_buffer(0, pipeline_system.vertex_buffer.slice(..));
            
            render_pass.draw(0..6, 0..1);
            
            
        }

        device_system.queue.submit(iter::once(encoder.finish()));
        //device_system.device.poll(wgpu::Maintain::Wait);
        output.present();

        Ok(())
    }
}


//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\

//
// Resizing
//

struct ResizeSystem {
    ctx: GeeseContextHandle<Self>,
}
impl GeeseSystem for ResizeSystem {

    const DEPENDENCIES: Dependencies = dependencies()
    .with::<PipelineSystem>()
    .with::<SurfaceSystem>()
    .with::<DeviceSystem>();

    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers().with(Self::resize);

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self { ctx }
    }
}

impl ResizeSystem{
    fn resize(&mut self, event: &on::WindowResized){
        

        let surface_system = self.ctx.get::<SurfaceSystem>();
        let surface = &surface_system.surface;
        let device_system = self.ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let /*mut*/ pipeline_system = self.ctx.get::<PipelineSystem>();
        let mut config = pipeline_system.config.clone();
        let mut size = pipeline_system.size;
        let new_size = event.physical_size;

        
        if new_size.width > 0 && new_size.height > 0 {
            
            size = new_size;
            config.width = new_size.width;
            config.height = new_size.height;
            surface.configure(&device, &config);
            
            
        }
        
    }
}


//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\

//
// Camera
//

struct Camera {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 2.0, 20.0],
            yaw: 45.0,
            pitch: 30.0,
            fov: 60.0,
        }
    }
}

impl Camera {
    fn new(position: [f32; 3], yaw: f32, pitch: f32, fov: f32) -> Self{
        Self {position, yaw, pitch, fov}
    }

    pub fn update_rotation(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.05; // Adjust this value to change mouse sensitivity

        // Update yaw and pitch with sensitivity scaling
        self.yaw -= delta_x * sensitivity;
        self.pitch -= delta_y * sensitivity;

        // Clamp pitch to avoid flipping at the poles
        self.pitch = self.pitch.clamp(-89.0, 89.0);
        self.yaw = (self.yaw + 360.0) % 360.0;
        // println!("Delta X: {}, Delta Y: {}", delta_x, delta_y);
        // println!("Updated Yaw: {}, Updated Pitch: {}", self.yaw, self.pitch);


    }

    fn convert_to_buffer(&mut self) -> CameraBuffer{
        //let yaw_rad = self.yaw.to_radians();
        //let pitch_rad = self.pitch.to_radians();

        // Compute the direction vector
        // println!("Normalized Direction: {:?}", direction);
        // println!("Magnitude: {}", direction.magnitude());

        //println!("Direction: {:?}", direction);
        //let (yaw, pitch) = direction_to_yaw_pitch(direction);
        //println!("yaw: {} pitch: {}", self.yaw, self.pitch);


        CameraBuffer {
            position: self.position,
            yaw: self.yaw,
            pitch: self.pitch,
            aspect: 16.0 / 9.0,
            fov: self.fov,
            padding3: 0.0,
        }
        

    }
}
struct CameraSystem {
    camera: Camera,
    camera_buffer: CameraBuffer,
    gpu_camera_buffer: Buffer,
    camera_bind_group: BindGroup,
    camera_bind_group_layout: BindGroupLayout,
    ctx: GeeseContextHandle<Self>,
}

impl GeeseSystem for CameraSystem {
    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers().with(Self::update_rotation);
    const DEPENDENCIES: Dependencies = dependencies()
    .with::<DeviceSystem>();
    
    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        let device = ctx.get::<DeviceSystem>().device.clone();
        let mut camera = Camera::default();

        let camera_buffer = camera.convert_to_buffer();
        let gpu_camera_buffer = camera_buffer.to_gpu_buffer(&device);

        let camera_bind_group_layout = //&camera_system.camera_bind_group_layout;
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }              
            ],
            label: Some("camera_bind_group_layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_camera_buffer.as_entire_binding()
                 /*wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &gpu_camera_buffer,
                    offset: 0,
                    size: None
                }),*/
            },],
            label: Some("Camera Bind Group"),

            /*
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dst_buffer.as_entire_binding(),
            }*/

            
        });

        Self { 
            camera_buffer,
            gpu_camera_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            camera,
            ctx,
        }
    }
}

impl CameraSystem {
    fn update_rotation(&mut self, event: &on::MouseMoved){
        let device_system = self.ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let queue = &device_system.queue;
        self.camera.update_rotation(event.delta_x, event.delta_y); // calculate direction 3D

        self.camera_buffer = self.camera.convert_to_buffer(); // Convert it into Camera Buffer form :)

        //println!("rotation: {} {} {}", self.camera_buffer.rotation[0],self.camera_buffer.rotation[1],self.camera_buffer.rotation[2]); // debug printing

        let binding = [self.camera_buffer];
        let buffer_data = bytemuck::cast_slice(&binding);
        queue.write_buffer(&self.gpu_camera_buffer, 0, buffer_data);
        
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("camera buffer encoder"),
        });
        

        queue.submit(std::iter::once(encoder.finish()));
        
    }

    fn update(&mut self){
        let device_system = self.ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let queue = &device_system.queue;
        self.camera_buffer = self.camera.convert_to_buffer();

        let binding = [self.camera_buffer];
        let buffer_data = bytemuck::cast_slice(&binding);
        queue.write_buffer(&self.gpu_camera_buffer, 0, buffer_data);
        
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("camera buffer encoder"),
        });
        

        queue.submit(std::iter::once(encoder.finish()));
    }

}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraBuffer {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    fov: f32,
    aspect: f32,
    padding3: f32,              // 4 bytes
}

impl CameraBuffer {
    /// Creates a GPU-ready buffer for the CameraBuffer.
    pub fn to_gpu_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        // Convert the CameraBuffer to a byte slice
        let buffer_data = bytemuck::bytes_of(self);

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: std::mem::size_of::<CameraBuffer>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        })


    }
}

pub fn direction_to_yaw_pitch(direction: Vector3<f32>) -> (f32, f32) {
    // Ensure the direction vector is normalized
    let direction = direction.normalize();

    // Calculate pitch (rotation around the X-axis)
    // asin gives us the angle in radians, convert to degrees
    let pitch = direction.y.asin().to_degrees();

    // Calculate yaw (rotation around the Y-axis)
    // atan2 gives us the angle in radians, convert to degrees
    let yaw = direction.z.atan2(direction.x).to_degrees();

    (yaw, pitch)
}

//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\

//
// Free Cam
//

struct FreeCamSystem {
    ctx: GeeseContextHandle<Self>,
}

impl FreeCamSystem {
    fn move_camera(&mut self, event: &on::KeyPressed){

        
        let delta_time = 0.016;
        let mut camera_system = self.ctx.get_mut::<CameraSystem>();
        let yaw = camera_system.camera.yaw;
        let pitch = camera_system.camera.pitch;
        let camera_position = camera_system.camera.position;
        let speed = 0.5;

        let (up, forward, right) = calculate_directions(yaw, pitch);


        let mut newpos: [f32; 3];
        


        if event.key == KeyCode::KeyA && event.state == ElementState::Pressed {
            let newpos: [f32; 3] = apply_motion(camera_position.into(), right, speed, delta_time, false).into();
            camera_system.camera.position = newpos.into();
            //println!("A key pressed: {:?}", event.key);
        }

        if event.key == KeyCode::KeyD {
            let newpos: [f32; 3]  = apply_motion(camera_position.into(), right, speed, delta_time, true).into();
            camera_system.camera.position = newpos.into();
            //println!("D key pressed: {:?}", event.key);
        }

        if event.key == KeyCode::KeyW {
            let newpos: [f32; 3]  = apply_motion(camera_position.into(), forward, speed, delta_time, false).into();
            camera_system.camera.position = newpos.into();
            //println!("W key pressed: {:?}", event.key);
        }

        if event.key == KeyCode::KeyS {
            let newpos: [f32; 3]  = apply_motion(camera_position.into(), forward, speed, delta_time, true).into();
            camera_system.camera.position = newpos.into();
            
            //println!("S key pressed: {:?}", event.key);
        }

        if event.key == KeyCode::KeyE && event.state == ElementState::Pressed {
            let newpos: [f32; 3] = apply_motion(camera_position.into(), up, speed, delta_time, false).into();
            camera_system.camera.position = newpos.into();
            //println!("A key pressed: {:?}", event.key);
        }

        if event.key == KeyCode::KeyQ && event.state == ElementState::Pressed {
            let newpos: [f32; 3] = apply_motion(camera_position.into(), up, speed, delta_time, true).into();
            camera_system.camera.position = newpos.into();
            //println!("A key pressed: {:?}", event.key);
        }


        camera_system.update();


    }

    
}


impl GeeseSystem for FreeCamSystem {
    const EVENT_HANDLERS: EventHandlers<Self> = event_handlers().with(Self::move_camera);
    const DEPENDENCIES: Dependencies = dependencies()
    .with::<Mut<CameraSystem>>();

    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        Self { ctx }
    }
}

use glam::{Vec3, Mat3};

fn calculate_directions(yaw: f32, pitch: f32) -> (Vec3, Vec3, Vec3) {
    // Convert yaw and pitch from degrees to radians
    let yaw_rad = yaw.to_radians();
    let pitch_rad = pitch.to_radians();

    // Forward direction
    let forward = Vec3::new(
        yaw_rad.cos() * pitch_rad.cos(),
        pitch_rad.sin(),
        yaw_rad.sin() * pitch_rad.cos(),
    )
    .normalize();

    // Right direction (cross product of forward and world up)
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let right = forward.cross(world_up).normalize();

    // Up direction (cross product of right and forward)
    let up = right.cross(forward).normalize();

    (up, forward, right)
}

fn apply_motion(
    position: Vec3,
    direction: Vec3,
    speed: f32,
    delta_time: f32,
    negative: bool,
) -> Vec3 {

    if negative == true {
        position - direction * speed// * delta_time
    } else {
        position + direction * speed// * delta_time
    }

    
}
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\
//|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\|\

struct WorldSystem {
    tree_manager: Tree64GpuManager,
    contree: Sparse64Tree,
}

impl GeeseSystem for WorldSystem {
    const DEPENDENCIES: Dependencies = dependencies()
    .with::<DeviceSystem>();


    fn new(ctx: GeeseContextHandle<Self>) -> Self {
        let device_system = &ctx.get::<DeviceSystem>();
        let device = &device_system.device;
        let queue = &device_system.queue;

        //let vox_path = "C:\\Users\\jonfr\\Documents\\Game Dev\\MagicaVoxel-0.99.7.1-win64\\MagicaVoxel-0.99.7.1-win64\\vox\\nature//mushrooms//bright_shroom (3).vox";
        //let contree = create_test_tree_from_vox(vox_path, 4);

        // Create a Perlin noise instance.
        let perlin = Perlin::new(1234);
        // Define the overall volume.
        let volume = AABB { 
            min: vec3(0.0, 0.0, 0.0), 
            max: vec3(64.0, 64.0, 64.0) 
        };
        // Create the tree and generate terrain.
        let mut contree = Sparse64Tree::new();
        contree.generate_terrain_sdf_noise_simd(volume, 4, perlin);

        add_vox_to_tree(vox_path, 4, 120, 30, 128, &mut contree);
        let mut tree_manager = Tree64GpuManager::new(&device, &contree);
        
        println!("contree nodes: {}", contree.nodes.len());


        Self {tree_manager, contree}
    }

}
