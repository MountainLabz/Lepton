// Add these to your Cargo.toml:
// slotmap = "1.0"
// bytemuck = "1.9"
// wgpu = "0.14"  // or your version
// cgmath = "0.18" // if used elsewhere

use cgmath::num_traits::pow;
use dot_vox::load;
use slotmap::{SlotMap, new_key_type};
use std::{collections::{HashMap, VecDeque}, fs::File, io::{BufReader, Read}, path::Path};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

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
    /// (Depth 0 is the root; deeper levels refine the tree.)
    pub fn insert(&mut self, x: u32, y: u32, z: u32, depth: usize, color: [u8; 3]) {
        let mut current_key = self.root;
        for current_depth in 0..depth {
            let child_index = self.compute_child_index(x, y, z, current_depth, depth);
            
            // Check if we need to create a new child
            let need_new_child;
            let rank;
            
            {  // Scope to limit the first mutable borrow
                let node = self.nodes.get_mut(current_key).unwrap();
                need_new_child = (node.child_mask & (1 << child_index)) == 0;
                
                // Calculate rank regardless of whether we need a new child
                rank = (node.child_mask & ((1 << child_index) - 1)).count_ones() as usize;
                
                // If we need a new child, update the mask now
                if need_new_child {
                    node.child_mask |= 1 << child_index;
                }
            }  // First borrow ends here
            
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
    // pub fn insert(&mut self, x: u32, y: u32, z: u32, depth: usize, color: [u8; 3]) {
    //     let mut current_key = self.root;
    //     for current_depth in 0..depth {
    //         let child_index = self.compute_child_index(x, y, z, current_depth, depth);
    //         // Get mutable access to the current node.
    //         let node = self.nodes.get_mut(current_key).unwrap();
    //         // Check if the bit for this child is set.
    //         if (node.child_mask & (1 << child_index)) == 0 {
    //             // Compute the "rank" – the number of set bits before this child index.
    //             let rank = (node.child_mask & ((1 << child_index) - 1)).count_ones() as usize;
    //             // Mark this child as present.
    //             node.child_mask |= 1 << child_index;
    //             // Create a new node and insert it into the slotmap.
    //             let new_key = self.nodes.insert(Node64::default());
    //             // Insert the new child's key in the proper position.
    //             node.children.insert(rank, new_key);
    //             current_key = new_key;
    //         } else {
    //             // Child already exists – look it up by computing its rank.
    //             let rank = (node.child_mask & ((1 << child_index) - 1)).count_ones() as usize;
    //             current_key = node.children[rank];
    //         }
    //     }
    //     // At the leaf node, set the voxel data and mark as a leaf.
    //     if let Some(leaf_node) = self.nodes.get_mut(current_key) {
    //         leaf_node.voxel_data = color;
    //         leaf_node.is_leaf = true;
    //     }
    // }
    
    /// Computes the child index from the voxel coordinates.
    /// This function is identical in spirit to your original implementation.
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
}

/// GPU-side node representation. This layout is designed to match your original
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


use noise::{Fbm, NoiseFn, Perlin};
// Test function to create a simple tree
// Test function to create a simple tree
pub fn create_test_tree() -> Sparse64Tree {
    let perlin = Perlin::new(1);

    let mut tree = Sparse64Tree::new();

    let depth = 4;
    let past_grid: i32 = pow(4, depth);
    let grid_size = past_grid as u32;
    let noise_size = 32;

    for x in 0..grid_size {
        for z in 0..grid_size {
            let val = ((perlin.get([(x as f64) / noise_size as f64, (z as f64) / noise_size as f64]) * noise_size as f64) + 10.0).abs() as u32;

            for y in 0..val {
                if y > 4 {
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

// fn generate_0_to_3() -> u32 {
//     let mut rng = rand::thread_rng();
//     rng.gen_range(0..=6) // Inclusive range from 0 to 3
// }

fn generate_0_to_255() -> u8 {
    let mut rng = rand::thread_rng();
    rng.gen_range(0..=255) as u8 // Inclusive range from 0 to 3
}
// // Example usage:
// /*
// pub fn setup_tree_for_gpu(device: &wgpu::Device, queue: &wgpu::Queue) -> Tree64GpuManager {
//     let mut gpu_manager = Tree64GpuManager::new(device);
//     let test_tree = create_test_tree();
//     gpu_manager.upload_tree(queue, &test_tree);
//     gpu_manager
// }
// */

// // Convert CPU node to GPU format
// fn convert_node_to_gpu(node: &Node64) -> GpuNode64 {

//     //println!("Converting node - raw color: {:?}", node.voxel_data);
    
//     let color = [
//         node.voxel_data[0] as f32 / 255.0,
//         node.voxel_data[1] as f32 / 255.0,
//         node.voxel_data[2] as f32 / 255.0,
//     ];

//     //println!("Normalized color: {:?}", color);

//     GpuNode64 {
//         child_mask_low: (node.child_mask & 0xFFFFFFFF) as u32,
//         child_mask_high: (node.child_mask >> 32) as u32,
//         child_ptr_and_leaf: node.child_ptr | if node.is_leaf() { 0x80000000 } else { 0 },
//         _padding: 0,
//         color,
//         _padding2: 0,
//     }
// }

// pub fn collect_nodes(tree: &Sparse64Tree, device: &wgpu::Device) -> Buffer{
//     let gpu_nodes: Vec<GpuNode64> = tree.nodes
//         .iter()
//         .map(|node| convert_node_to_gpu(node))
//         .collect();

//     let node_slice = bytemuck::cast_slice(&gpu_nodes);

//     let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("Indices Buffer"),
//         contents: node_slice,
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//     });

//     return node_buffer;
// }

// fn generate_and_check() -> (u32, bool) {
//     // Create a random number generator.  Using `thread_rng()` is usually sufficient.
//     let mut rng = rand::thread_rng();

//     // Generate a random number between 1 and 100 (inclusive).
//     let number = rng.gen_range(1..=100); // Note the ..= for inclusive range

//     // Check if the number is greater than 50.
//     let is_greater_than_50 = number > 50;

//     // Return the number and the boolean result as a tuple.
//     (number, is_greater_than_50)
// }