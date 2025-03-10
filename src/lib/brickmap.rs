/*brickmap to arrays problem solution:

create an array of indices like so:
indices = [i32; 32,768]
Then cycle through the brickmaps and for every brick in the brickmap, calculate the 3D position of the index in the total array of indices
and place the index into that slot of the indices array. at the same time appending the if solid brick to the brick array.
*/








use bytemuck::Zeroable;
use bytemuck::Pod;
//dot vox stuff:
use dot_vox::DotVoxData;
use dot_vox::load;
use dot_vox::Color;
use dot_vox::Size;
use geese::Mut;
use geese::SystemRef;
//geese
use geese::{
    dependencies, Dependencies, EventHandlers, EventQueue,
    GeeseContext, GeeseContextHandle, GeeseSystem, event_handlers,
};
use std::sync::{Arc, Mutex};
use std::{default, iter, usize};
//wgpu
use wgpu::core::device;
use wgpu::util::DeviceExt;
use wgpu::{Adapter, BindGroup, BindGroupLayout, Buffer, Device, Instance, PipelineCompilationOptions, Queue, RenderPipeline, Surface, SurfaceConfiguration, Texture, TextureView};
use wgpu::core::device::queue;



//
// Voxels
//

pub struct ChunkWorld{
    pub brickmaps: Vec<BrickMap>,
    pub indices_array: Vec<i32>,
    pub brick_array: Vec<Brick>,
}

impl ChunkWorld {
    pub fn new(size: usize) -> Self{
        let b_size = size * 16; // calculate the amount of bricks based upon the brickmaps instead of the pure brick count.
        let total_size = b_size * b_size * b_size;
        Self{ brickmaps: Vec::new(), indices_array: vec![-1; total_size], brick_array: Vec::new()}
    }

    pub fn create_world(&mut self, world_size: usize) {
        let brickmaps_total = world_size * world_size * world_size;
        for x in 0..world_size {
            for y in 0..world_size {
                for z in 0..world_size {
                    self.brickmaps.push(BrickMap::new(16, [x as i32,y as i32,z as i32]));
                }
            }
            
        }
    }

    pub fn populate_brickmaps(&mut self){
        for bm in self.brickmaps.iter_mut() {
            bm.generate_map();
            //print!("Generated Brickmap");

        }
    }


    // this function (created by claude Sonnet) extracts the bricks and indices into arrays.
    // fn extract_to_arrays(&mut self) {
    //     // First, calculate the total world dimensions in bricks
    //     let (min_pos, max_pos) = self.get_world_bounds();
    //     let world_size = (max_pos - min_pos + IVec3::ONE) * 16; // Convert to brick coordinates
        
    //     let total_bricks = (world_size.x * world_size.y * world_size.z) as usize;
    //     let mut brick_array = vec![None; total_bricks];
    //     let mut index_array = vec![0u32; total_bricks];
        
    //     // For each brickmap in the world
    //     for (brickmap_pos, brickmap) in &self.brickmaps {
    //         // Calculate base offset for this brickmap
    //         let base_offset = self.calculate_brickmap_offset(brickmap_pos, min_pos);
            
    //         // For each potential brick position in the brickmap
    //         for x in 0..16 {
    //             for y in 0..16 {
    //                 for z in 0..16 {
    //                     let local_idx = (x * 16 * 16 + y * 16 + z) as usize;
                        
    //                     // If there's a brick at this position
    //                     if let Some(brick_idx) = brickmap.indices[local_idx] {
    //                         if let Some(brick) = brickmap.bricks.get(&brick_idx) {
    //                             // Calculate global position
    //                             let global_pos = IVec3::new(
    //                                 base_offset.x + x,
    //                                 base_offset.y + y,
    //                                 base_offset.z + z
    //                             );
                                
    //                             // Convert 3D position to flat array index
    //                             let array_idx = self.position_to_index(global_pos, world_size);
                                
    //                             // Store brick and index
    //                             brick_array[array_idx] = Some(brick.clone());
    //                             index_array[array_idx] = brick_idx as u32;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
        
    //     ExtractedWorld {
    //         brick_array,
    //         index_array,
    //         dimensions: world_size,
    //     }
    // }
    //// this function was also written by Sonnet
    // fn position_to_index(&self, pos: IVec3, dimensions: IVec3) -> usize {
    //     (pos.x * dimensions.y * dimensions.z + pos.y * dimensions.z + pos.z) as usize
    // }


    pub fn collect_bricks(&mut self, world_size: usize){
        //let world_array_size = world_size * world_size * world_size;
        //let world_array_dim = (world_size as f32).cbrt() as usize;

        let brick_world_len = world_size * 16;
        let brick_array_size = brick_world_len * brick_world_len * brick_world_len;
        self.brick_array.clear();
        self.indices_array = vec![-1; brick_array_size];

        

        for brickmap in self.brickmaps.iter() {
            for x in 0..16{
                for y in 0..16{
                    for z in 0..16{
                        let index = calculate_flat_index(x, y, z, 16);
                        let brick_pos = [
                            brickmap.position[0] * 16 + x,
                            brickmap.position[1] * 16 + y,
                            brickmap.position[2] * 16 + z
                        ];
                        let global_index = calculate_flat_index(brick_pos[0], brick_pos[1], brick_pos[2], brick_world_len as i32);
                        if brickmap.indices[index] != -1{
                            self.indices_array[global_index] = self.brick_array.len() as i32;
                            self.brick_array.push(Brick::new());
                        }
                    }
                }
            }
        }
    }

    pub fn claude_collect_bricks(&mut self, world_size: usize) {
        let brick_world_len: usize = world_size * 16; // Total size in bricks
        let brick_array_size = brick_world_len * brick_world_len * brick_world_len;
        
        self.brick_array.clear();
        self.indices_array = vec![-1; brick_array_size];
    
        for brickmap in self.brickmaps.iter() {
            for x in 0..16 {
                for y in 0..16 {
                    for z in 0..16 {
                        let local_index = x * 16 * 16 + y * 16 + z;
                        
                        if brickmap.indices[local_index] != -1 {
                            // Calculate global brick position
                            let brick_pos = [
                                brickmap.position[0] * 16 + (x as i32),
                                brickmap.position[1] * 16 + (y as i32),
                                brickmap.position[2] * 16 + (z as i32)
                            ];
                            
                            // Calculate global index using the brick_world_len
                            let global_index = calculate_flat_index(
                                brick_pos[0], 
                                brick_pos[1], 
                                brick_pos[2], 
                                brick_world_len as i32
                            );
                            
                            //if global_index >= 0 && global_index < brick_array_size {
                            self.indices_array[global_index as usize] = self.brick_array.len() as i32;
                            self.brick_array.push(brickmap.bricks[brickmap.indices[local_index - 1] as usize].clone());
                            //}
                        }
                    }
                }
            }
        }
    }

    // pub fn collect_bricks_chat_gpt(&mut self, world_size: usize) {
    //     self.brick_array.clear();
    //     self.indices_array.clear();

    //     let brickmap_size = world_size * world_size * world_size;
    //     let brickmap_dim = (world_size as f32).cbrt() as usize;

    //     for z in 0..brickmap_dim {
    //         for y in 0..brickmap_dim {
    //             for x in 0..brickmap_dim {
    //                 // Calculate the brickmap index in the flat array
    //                 let brickmap_index = x + y * brickmap_dim + z * brickmap_dim * brickmap_dim;
    //                 let brickmap = &self.brickmaps[brickmap_index];

    //                 for (i, &brick_index) in brickmap.indices.iter().enumerate() {
    //                     let local_x = i % 16;
    //                     let local_y = (i / 16) % 16;
    //                     let local_z = i / (16 * 16);

    //                     let global_x = x * 16 + local_x;
    //                     let global_y = y * 16 + local_y;
    //                     let global_z = z * 16 + local_z;

    //                     // If there's a valid brick at this position
    //                     if brick_index >= 0 {
    //                         let brick = brickmap.bricks[brick_index as usize - 1];
    //                         self.brick_array.push(brick);

    //                         // Compute the flat index based on global 3D position
    //                         let flat_index = global_x
    //                             + global_y * (brickmap_dim * 16)
    //                             + global_z * (brickmap_dim * 16) * (brickmap_dim * 16);
    //                         self.indices_array.push(flat_index as i32);
    //                     } else if brick_index == -1 {
    //                         self.indices_array.push(-1);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}



pub struct GPUChunk {
    pub vox_world_bind_group: BindGroup,
    pub vox_world_bind_group_layout: BindGroupLayout,
    pub indices_buffer: wgpu::Buffer,
    pub brick_array_buffer: wgpu::Buffer,
}

impl GPUChunk {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, world: &ChunkWorld) -> Self {

        let (brick_array_buffer, indices_buffer) = upload_to_gpu(device, queue, &world.brick_array, &world.indices_array);


        // Borrow brick and index data directly from the world.
        // let brick_array_buffer = upload_single_buffer_to_gpu(
        //     device,
        //     queue,
        //     &world.brick_array,
        //     wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //     "Brick Array Buffer",
        // );

        // let indices_buffer = upload_single_buffer_to_gpu(
        //     device,
        //     queue,
        //     &world.indices_array,
        //     wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //     "Indices Buffer",
        // );

        // Create bind group layout (example setup).
        let vox_world_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // Binding for brick buffer.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding for index buffer.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("Voxel World Bind Group Layout"),
        });

        // Create bind group.
        let vox_world_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &vox_world_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: brick_array_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indices_buffer.as_entire_binding(),
                },
            ],
            label: Some("Voxel World Bind Group"),
        });

        Self {
            vox_world_bind_group,
            vox_world_bind_group_layout,
            indices_buffer,
            brick_array_buffer,
        }
    }
}






//
// structs
//

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, Zeroable, Pod, PartialEq, Eq, Hash)]
pub struct Voxel {
    pub material: u32,  // Material ID
}

impl Voxel {
    pub fn new(material: u32) -> Self {
        Self { material }
    }
}

/// A Brick containing an array of voxels and a flag for whether it contains data.
#[repr(C, align(8))]
#[derive(Debug, Clone, Zeroable, Pod, Copy, PartialEq, Eq, Hash)]
pub struct Brick {
    //pub voxels: Vec<Voxel>,
    pub voxels: [Voxel; 4096],
    //pub padding: [u8; 7],
}

use noise::{NoiseFn, Perlin, Seedable};
impl Brick{
    fn new() -> Self{
        Self{
            voxels: [Voxel::new(0); 4096],
            //voxels: Vec::new(),
        }
    }

}
use rand::prelude::*;
pub struct BrickMap {
    pub bricks: Vec<Brick>, // A flat array of brick data, not indexable do to its sparse nature.
    pub indices: Vec<i32>, // Aka the bitmap but its not bits lol.
    pub position: [i32; 3],
}

impl BrickMap{
    fn new(size: usize, position: [i32; 3]) -> Self{
        let new_size = size * size * size;
        let mut bricks: Vec<Brick> = Vec::new();
        let mut indices: Vec<i32> = vec![-1; new_size];
        

        let mut rng = thread_rng();

        
        // for i in 0..new_size {
        //     let should_gen_brick = rng.gen_bool(0.5);
        //     if should_gen_brick == true{
        //         bricks.push(Brick::new());
        //         indices[i] = bricks.len().try_into().unwrap();
        //         //indices.push(bricks.len().try_into().unwrap());
        //     }
            
        // }
        
        for x in 0..size {
            for y in 0..size {
                for z in 0..size {
                    if y > 1 {
                        let index = calculate_flat_index(x as i32, y as i32, z as i32, size as i32) as usize;
                        bricks.push(Brick::new());
                        indices[index] = bricks.len().try_into().unwrap();
                    }
                }
            }
        }

        Self { bricks, indices, position: position }
    }
    // !
    // this needs the indices list implementation, otherwise its useless lol.
    // !
    fn generate_map(&mut self){
        // let mut rng = thread_rng();
        // for brick in self.bricks.iter_mut() {
            
        //     for mut voxel in brick.voxels {
        //         let new_vox = Voxel::new(rng.gen_range(0..3));
        //         voxel = new_vox;
        //     }
            
        // }
    }
}

//
// Functions
//

pub fn generate_noise(x: f64, y: f64, z: f64) -> f64{
    let perlin = Perlin::new(1);
    let val: f64 = perlin.get([x, y, z]);
    return val;
}

use bytemuck::cast_slice;

pub fn upload_to_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bricks: &Vec<Brick>,      // Prepared brick buffer
    indices: &Vec<i32>,       // Prepared indices buffer
) -> (wgpu::Buffer, wgpu::Buffer) {
    // Convert Vec<Brick> to a byte slice
    let bricks_bytes = cast_slice(bricks);

    // Convert Vec<i32> to a byte slice
    let indices_bytes = cast_slice(indices);

    // Create the buffer for bricks
    let brick_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Brick Buffer"),
        contents: bricks_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create the buffer for indices
    let indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Indices Buffer"),
        contents: indices_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Return the GPU buffers
    (brick_buffer, indices_buffer)
}

//
// misc functions
//

fn calculate_flat_index(x: i32, y: i32, z: i32, dim: i32) -> usize {
    (x * dim * dim + y * dim + z) as usize
}
