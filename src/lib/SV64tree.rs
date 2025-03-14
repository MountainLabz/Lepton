use cgmath::num_traits::pow;
use cgmath::{vec3, ElementWise, Vector3};
use dot_vox::load;
use simdnoise::*;
use slotmap::{SlotMap, new_key_type};
use std::{collections::{HashMap, VecDeque}, fs::File, io::{BufReader, Read}, path::Path};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wide::f32x4;  // Using the wide crate for SIMD arithmetic

#[derive(Clone, Copy)]
pub struct AABB {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

impl AABB {
    /// Returns the 8 corners of the box.
    pub fn corners(&self) -> [Vector3<f32>; 8] {
         [
            self.min,
            Vector3::new(self.max.x, self.min.y, self.min.z),
            Vector3::new(self.min.x, self.max.y, self.min.z),
            Vector3::new(self.min.x, self.min.y, self.max.z),
            Vector3::new(self.max.x, self.max.y, self.min.z),
            Vector3::new(self.max.x, self.min.y, self.max.z),
            Vector3::new(self.min.x, self.max.y, self.max.z),
            self.max,
         ]
    }
}

new_key_type! {
    /// A stable key for each node in the tree.
    pub struct NodeKey;
}

/// A single node in the sparse voxel tree.
/// 
/// Instead of a flat pointer (u32) into a contiguous array, we now store
/// a vector of child keys. The `child_mask` is kept to preserve the
/// same “sparse” child ordering as before.
#[derive(Debug)]
pub struct Node64 {
    /// 64-bit bitmask indicating which child indices are present.
    child_mask: u64,
    /// The children, stored in the order implied by the mask.
    children: Vec<NodeKey>,
    /// The voxel’s RGB color data.
    voxel_data: [u8; 3],
    /// Whether this node is a leaf.
    is_leaf: bool,
}

impl Default for Node64 {
    fn default() -> Self {
        Self {
            child_mask: 0,
            children: Vec::new(),
            voxel_data: [0, 0, 0],
            is_leaf: false,
        }
    }
}

/// The sparse voxel tree using a slotmap to hold nodes.
pub struct Sparse64Tree {
    /// The nodes stored in a slotmap for stable keys.
    pub nodes: SlotMap<NodeKey, Node64>,
    /// The root node key.
    pub root: NodeKey,
}

impl Sparse64Tree {
    /// Creates a new tree with a default root node.
    pub fn new() -> Self {
        let mut nodes = SlotMap::with_key();
        let root = nodes.insert(Node64::default());
        Self { nodes, root }
    }
    
    /// Inserts a voxel at the given (x, y, z) and depth with the provided color.
    pub fn insert(&mut self, x: u32, y: u32, z: u32, depth: usize, color: [u8; 3]) {
        let mut current_key = self.root;
        for current_depth in 0..depth {
            let child_index = self.compute_child_index(x, y, z, current_depth, depth);
            
            // Check if we need to create a new child
            let need_new_child;
            let rank;
            
            {
                let node = self.nodes.get_mut(current_key).unwrap();
                need_new_child = (node.child_mask & (1 << child_index)) == 0;
                
                // Calculate rank regardless of whether we need a new child
                rank = (node.child_mask & ((1 << child_index) - 1)).count_ones() as usize;
                
                // If we need a new child, update the mask now
                if need_new_child {
                    node.child_mask |= 1 << child_index;
                }
            }
            
            if need_new_child {
                // Now we can modify self.nodes again to insert a new node
                let new_key = self.nodes.insert(Node64::default());
                
                // And now get the node again to update children
                let node = self.nodes.get_mut(current_key).unwrap();
                node.children.insert(rank, new_key);
                current_key = new_key;
            } else {
                // Get the existing child's key
                let child_key = {
                    let node = self.nodes.get(current_key).unwrap();  // Immutable borrow is fine here
                    node.children[rank]
                };
                current_key = child_key;
            }
        }
        
        // At the leaf node, set the voxel data and mark as a leaf
        if let Some(leaf_node) = self.nodes.get_mut(current_key) {
            leaf_node.voxel_data = color;
            leaf_node.is_leaf = true;
        }
    }
    
    /// Computes the child index from the voxel coordinates.
    fn compute_child_index(&self, x: u32, y: u32, z: u32, depth: usize, max_depth: usize) -> usize {
        let shift = (max_depth - depth - 1) * 2; // Reverse the shift order.
        let x_part = ((x >> shift) & 0b11) as usize;
        let y_part = ((y >> shift) & 0b11) as usize;
        let z_part = ((z >> shift) & 0b11) as usize;
        (z_part << 4) | (y_part << 2) | x_part
    }
    
    /// Flattens the slotmap-based tree into a contiguous byte vector for GPU upload.
    /// A breadth-first traversal assigns contiguous indices to nodes and computes each
    /// node’s child pointer (the index of its first child, if any).
    pub fn flatten(&self) -> Vec<u8> {
        let mut flattened_nodes = Vec::new();
        let mut key_to_index = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back(self.root);
        while let Some(key) = queue.pop_front() {
            let index = flattened_nodes.len();
            key_to_index.insert(key, index);
            let node = self.nodes.get(key).unwrap();
            flattened_nodes.push((key, node));
            // Enqueue children in the order stored in the node.
            for &child_key in &node.children {
                queue.push_back(child_key);
            }
        }
        
        // Convert each node into its GPU-friendly representation.
        let gpu_nodes: Vec<GpuNode64> = flattened_nodes
            .iter()
            .map(|(_key, node)| {
                // For non-leaf nodes, the child pointer is the flattened index of the first child.
                let child_ptr = if !node.children.is_empty() {
                    *key_to_index.get(&node.children[0]).unwrap() as u32
                } else {
                    0
                };
                // The highest bit in the child_ptr field marks the node as a leaf.
                let leaf_flag = if node.is_leaf { 0x8000_0000 } else { 0 };
                let child_ptr_and_leaf = child_ptr | leaf_flag;
                let child_mask_low = (node.child_mask & 0xFFFF_FFFF) as u32;
                let child_mask_high = (node.child_mask >> 32) as u32;
                // Convert the stored voxel color (u8) to f32 in the range [0.0, 1.0].
                let color = [
                    node.voxel_data[0] as f32 / 255.0,
                    node.voxel_data[1] as f32 / 255.0,
                    node.voxel_data[2] as f32 / 255.0,
                ];
                GpuNode64 {
                    child_mask_low,
                    child_mask_high,
                    child_ptr_and_leaf,
                    _padding: 0,
                    color,
                    _padding2: 0,
                }
            })
            .collect();
        
        // Cast the GPU node slice to bytes.
        bytemuck::cast_slice(&gpu_nodes).to_vec()
    }




    /// Generates terrain using the SIMD-accelerated noise-based SDF.
    /// A noise cache is used to avoid duplicate noise evaluations.
    pub fn generate_terrain_sdf_noise_simd(&mut self, aabb: AABB, max_depth: usize, noise: Perlin) {
        let mut noise_cache = HashMap::new();
        self.subdivide_node_noise_simd(self.root, aabb, 0, max_depth, &noise, &mut noise_cache);
    }

    /// Recursively subdivides the volume based on SDF values computed with SIMD,
    /// using the noise cache to reduce repeated noise evaluations.
    fn subdivide_node_noise_simd(
        &mut self,
        node_key: NodeKey,
        aabb: AABB,
        depth: usize,
        max_depth: usize,
        noise: &Perlin,
        cache: &mut HashMap<(i32, i32), f32>,
    ) {
        let corners = aabb.corners();
        let sdf_values = eval_corners_simd(noise, &corners, cache);
        let center_sdf = sdf_values.iter().sum::<f32>() / 8.0;

        // Skip cell if completely outside.
        if sdf_values.iter().all(|&v| v >= 0.0) {
            return;
        }
        // Mark cell as solid if completely inside.
        if sdf_values.iter().all(|&v| v < 0.0) {
            if let Some(node) = self.nodes.get_mut(node_key) {
                node.is_leaf = true;
                node.voxel_data = [34 + (generate_0_to_255() / 5), 139, 34 + (generate_0_to_255() / 5)]; // terrain green
            }
            return;
        }

        // If not at maximum depth, subdivide into a 4×4×4 grid.
        if depth < max_depth {
            let cell_size = (aabb.max - aabb.min) / 4.0;
            for cz in 0..4 {
                for cy in 0..4 {
                    for cx in 0..4 {
                        let offset = vec3(cx as f32, cy as f32, cz as f32);
                        let child_min = aabb.min + offset.mul_element_wise(cell_size);
                        let child_max = child_min + cell_size;
                        let child_aabb = AABB { min: child_min, max: child_max };

                        let child_corners = child_aabb.corners();
                        let child_sdf_values = eval_corners_simd(noise, &child_corners, cache);
                        if child_sdf_values.iter().all(|&v| v >= 0.0) {
                            continue;
                        }

                        let child_index = (cz << 4) | (cy << 2) | cx;
                        let new_child_key = self.nodes.insert(Node64::default());
                        {
                            let parent = self.nodes.get_mut(node_key).unwrap();
                            let rank = (parent.child_mask & ((1 << child_index) - 1)).count_ones() as usize;
                            if parent.child_mask & (1 << child_index) == 0 {
                                parent.child_mask |= 1 << child_index;
                                parent.children.insert(rank, new_child_key);
                            }
                        }
                        self.subdivide_node_noise_simd(new_child_key, child_aabb, depth + 1, max_depth, noise, cache);
                    }
                }
            }
        } else {
            // At maximum depth, mark as solid if the average (center) SDF is below zero.
            if center_sdf < 0.0 {
                if let Some(node) = self.nodes.get_mut(node_key) {
                    node.is_leaf = true;
                    node.voxel_data = [34 + (generate_0_to_255() / 5), 139, 34 + (generate_0_to_255() / 5)];
                }
            }
        }
    }





}


/// A helper that caches noise values to avoid repeated expensive evaluations.
/// quantize the (x, z) coordinates.
fn get_noise_value(
    noise: &Perlin,
    p: &Vector3<f32>,
    cache: &mut HashMap<(i32, i32), f32>,
) -> f32 {
    
    // a quantization factor of 20.0 (1/0.05 = 20) so that integer coordinates match.
    let key = (((p.x * 20.0).floor()) as i32, ((p.z * 20.0).floor()) as i32);
    if let Some(&value) = cache.get(&key) {
        value
    } else {
        let value = noise.get([p.x as f64 * 0.05, p.z as f64 * 0.05]) as f32;
        cache.insert(key, value);
        value
    }
}

/// Evaluates the SDF at 8 corners using SIMD for the arithmetic part,
/// and uses a cache to avoid duplicate noise evaluations.
fn eval_corners_simd(
    noise: &Perlin,
    corners: &[Vector3<f32>; 8],
    cache: &mut HashMap<(i32, i32), f32>,
) -> [f32; 8] {
    let mut result = [0.0f32; 8];
    // Process the 8 corners in two batches of 4 lanes.
    for i in 0..2 {
        let start = i * 4;
        let xs = f32x4::from([
            corners[start].x,
            corners[start + 1].x,
            corners[start + 2].x,
            corners[start + 3].x,
        ]);
        let ys = f32x4::from([
            corners[start].y,
            corners[start + 1].y,
            corners[start + 2].y,
            corners[start + 3].y,
        ]);
        let raw_noise_arr = [
            get_noise_value(noise, &corners[start], cache),
            get_noise_value(noise, &corners[start + 1], cache),
            get_noise_value(noise, &corners[start + 2], cache),
            get_noise_value(noise, &corners[start + 3], cache),
        ];
        let raw_noise = f32x4::from(raw_noise_arr);

        // Smooth the noise: smooth = raw^2 * (3 - 2 * raw)
        let smooth_noise = raw_noise * raw_noise * (f32x4::splat(3.0) - f32x4::splat(2.0) * raw_noise);
        let base_height = f32x4::splat(10.0);
        let amplitude = f32x4::splat(5.0);
        let interpolated_height = base_height + smooth_noise * amplitude;
        // The SDF is the difference between the y coordinate and the interpolated height.
        let mut sdf = ys - interpolated_height;

        let sdf_arr = sdf.as_array_mut();
        result[start] = sdf_arr[0];
        result[start + 1] = sdf_arr[1];
        result[start + 2] = sdf_arr[2];
        result[start + 3] = sdf_arr[3];
    }
    result
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


/// GPU-side node representation.
/// structure (child_mask split into two u32, a 32-bit pointer with a leaf flag, and color).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuNode64 {
    pub child_mask_low: u32,
    pub child_mask_high: u32,
    pub child_ptr_and_leaf: u32,
    pub _padding: u32,
    pub color: [f32; 3],
    pub _padding2: u32, // For 16-byte alignment.
}

/// Manager for uploading and binding the tree to the GPU.
pub struct Tree64GpuManager {
    node_buffer: wgpu::Buffer,
    num_nodes: u32,
    pub contree_bind_group: wgpu::BindGroup,
    pub contree_bind_group_layout: wgpu::BindGroupLayout,
}

impl Tree64GpuManager {
    /// Creates a new GPU manager by flattening the tree and uploading it to a buffer.
    pub fn new(device: &wgpu::Device, contree: &Sparse64Tree) -> Self {
        let node_buffer = Self::collect_nodes(contree, device);
        let contree_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Binding for the tree buffer.
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
                ],
                label: Some("Contree Bind Group Layout"),
            });
        
        // Create the bind group.
        let contree_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &contree_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: node_buffer.as_entire_binding(),
            }],
            label: Some("Contree Bind Group"),
        });
        
        Self {
            node_buffer,
            num_nodes: contree.nodes.len() as u32,
            contree_bind_group,
            contree_bind_group_layout,
        }
    }
    
    /// Flattens the tree and creates a GPU buffer for it.
    pub fn collect_nodes(tree: &Sparse64Tree, device: &wgpu::Device) -> wgpu::Buffer {
        let node_data = tree.flatten();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Node Buffer"),
            contents: &node_data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }
    
    /// Reuploads an updated tree to the GPU.
    pub fn upload_tree(&mut self, queue: &wgpu::Queue, tree: &Sparse64Tree) {
        let node_data = tree.flatten();
        self.num_nodes = tree.nodes.len() as u32;
        queue.write_buffer(&self.node_buffer, 0, &node_data);
    }
    
    /// Returns a reference to the underlying node buffer.
    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.node_buffer
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



use noise::{Fbm, NoiseFn, Perlin};
// Test function to create a simple tree
pub fn create_test_tree() -> Sparse64Tree {
    let perlin = Perlin::new(1);

    let mut tree = Sparse64Tree::new();

    let depth = 4;
    let past_grid: i32 = pow(4, depth);
    let grid_size = past_grid as u32;
    let noise_size = 128;

    for x in 0..grid_size {
        for z in 0..grid_size {
            let val = ((perlin.get([(x as f64) / noise_size as f64, (z as f64) / noise_size as f64]) * 10.0/* noise_size as f64*/) + 50.0).abs() as u32;
            for y in 0..val {
                if y > (val / 2) {
                    tree.insert(x, y, z, depth as usize, [0, 121 + (generate_0_to_255() / 5), 40]);
                } else {
                    tree.insert(x, y, z, depth as usize, [121 + (generate_0_to_255() / 15), 60, 0]);
                }
            }
        }
    }

    tree
}

// pub fn create_test_tree_from_voxel_file(filepath: &str, depth: usize) -> Sparse64Tree {
//     let mut tree = Sparse64Tree::new();

//     let path = Path::new(filepath);
//     let file = match File::open(&path) {
//         Err(why) => {
//             eprintln!("Couldn't open {}: {}", filepath, why);
//             return tree; // Return an empty tree if file opening fails.
//         }
//         Ok(file) => file,
//     };

//     let mut reader = BufReader::new(file);
//     let mut buffer = [0u8; 4]; // x,y,z,color

//     while reader.read_exact(&mut buffer).is_ok() {
//         let x = buffer[0] as u32;
//         let y = buffer[1] as u32;
//         let z = buffer[2] as u32;
//         let color = [buffer[3], generate_0_to_255(), generate_0_to_255()]; // Randomize 2 color components.

//         tree.insert(x, y, z, depth, color);
//     }

//     tree
// }




pub fn add_vox_to_tree(file_path: &str, depth: usize, offsetx: u32, offsety: u32, offsetz: u32, tree: &mut Sparse64Tree){

    // Load the .vox file
    let vox_data = match load(Path::new(file_path).to_str().expect("msg")) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load .vox file: {}", e);
            return;
        }
    };

    // Iterate through all models in the .vox file
    for model in &vox_data.models {
        let model_offset = (0, 0, 0); // Default offset (since dot_vox doesn't provide one directly)

        for voxel in &model.voxels {
            let x = voxel.x as u32 + model_offset.0;
            let y = voxel.z as u32 + model_offset.1;
            let z = voxel.y as u32 + model_offset.2;
            let color_index = voxel.i as usize;

            // Get color from the palette, fallback to white if out of bounds
            let color = if color_index < vox_data.palette.len() {
                let c = vox_data.palette[color_index];
                [c.r, c.g, c.b]
            } else {
                [255, 255, 255] // Default to white if index is out of bounds
            };

            // Insert voxel into the tree with the provided depth
            tree.insert(x + offsetx, y + offsety, z + offsetz, depth, color);
        }
    }
}

pub fn create_test_tree_from_vox(file_path: &str, depth: usize) -> Sparse64Tree {
    let mut tree = Sparse64Tree::new();

    // Load the .vox file
    let vox_data = match load(Path::new(file_path).to_str().expect("msg")) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load .vox file: {}", e);
            return tree;
        }
    };

    // Iterate through all models in the .vox file
    for model in &vox_data.models {
        let model_offset = (0, 0, 0); // Default offset (since dot_vox doesn't provide one directly)

        for voxel in &model.voxels {
            let x = voxel.x as u32 + model_offset.0;
            let y = voxel.z as u32 + model_offset.1;
            let z = voxel.y as u32 + model_offset.2;
            let color_index = voxel.i as usize;

            // Get color from the palette, fallback to white if out of bounds
            let color = if color_index < vox_data.palette.len() {
                let c = vox_data.palette[color_index];
                [c.r, c.g, c.b]
            } else {
                [255, 255, 255] // Default to white if index is out of bounds
            };

            // Insert voxel into the tree with the provided depth
            tree.insert(x, y, z, depth, color);
        }
    }

    tree
}


use rand::{random, Rng};

fn generate_0_to_255() -> u8 {
    let mut rng = rand::thread_rng();
    rng.gen_range(0..=255) as u8 // Inclusive range from 0 to 3
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

/// Manages multiple Sparse64Trees in a single GPU buffer.
pub struct TreeMemoryManager {
    buffer: wgpu::Buffer,
    root_node_buffer: wgpu::Buffer, // Stores root node references
    capacity: usize,
    allocations: HashMap<u32, usize>, // Maps tree ID to buffer offset
    free_slots: Vec<usize>, // Tracks free slots
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub root_bind_group: wgpu::BindGroup,
    pub root_bind_group_layout: wgpu::BindGroupLayout,
}

impl TreeMemoryManager {
    /// Creates a new memory manager with a given capacity.
    pub fn new(device: &wgpu::Device, max_trees: usize, tree_size: usize) -> Self {
        let buffer_size = max_trees * tree_size;
        let root_buffer_size = max_trees * std::mem::size_of::<u32>();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Voxel Tree Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let root_node_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tree Root Node Buffer"),
            size: root_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tree Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tree Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        let root_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Root Node Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let root_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Root Node Bind Group"),
            layout: &root_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: root_node_buffer.as_entire_binding(),
            }],
        });

        let free_slots = (0..max_trees).map(|i| i * tree_size).collect();

        Self {
            buffer,
            root_node_buffer,
            capacity: max_trees,
            allocations: HashMap::new(),
            free_slots,
            bind_group,
            bind_group_layout,
            root_bind_group,
            root_bind_group_layout,
        }
    }

    /// Allocates space for a new tree, uploads data, and stores the root node reference.
    pub fn upload_tree(&mut self, queue: &wgpu::Queue, tree_id: u32, tree_data: &[u8], root_node_offset: u32) {
        if let Some(&offset) = self.allocations.get(&tree_id) {
            queue.write_buffer(&self.buffer, offset as u64, tree_data);
            self.update_root_reference(queue, tree_id, root_node_offset);
        } else if let Some(offset) = self.free_slots.pop() {
            self.allocations.insert(tree_id, offset);
            queue.write_buffer(&self.buffer, offset as u64, tree_data);
            self.update_root_reference(queue, tree_id, root_node_offset);
        } else {
            panic!("No available slots for new trees!");
        }
    }

    /// Updates the root node reference for a specific tree.
    pub fn update_root_reference(&self, queue: &wgpu::Queue, tree_id: u32, root_node_offset: u32) {
        let root_index = tree_id as usize * std::mem::size_of::<u32>();
        queue.write_buffer(&self.root_node_buffer, root_index as u64, bytemuck::bytes_of(&root_node_offset));
    }

    /// Removes a tree from the GPU buffer, marking its slot as free.
    pub fn remove_tree(&mut self, tree_id: u32) {
        if let Some(offset) = self.allocations.remove(&tree_id) {
            self.free_slots.push(offset);
        }
    }

    /// Returns the buffer reference.
    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Returns the root node buffer reference.
    pub fn get_root_node_buffer(&self) -> &wgpu::Buffer {
        &self.root_node_buffer
    }
}