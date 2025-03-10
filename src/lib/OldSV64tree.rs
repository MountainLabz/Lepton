#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct Node64 {
    child_mask: u64,      // 64-bit child presence mask
    child_ptr: u32,       // 31 bits for pointer, highest bit for is_leaf
    voxel_data: [u8; 3],  // RGB color data
}

impl Node64 {
    // Helper methods to get/set leaf status and pointer
    fn is_leaf(&self) -> bool {
        (self.child_ptr & 0x80000000) != 0
    }

    fn set_leaf(&mut self, is_leaf: bool) {
        if is_leaf {
            self.child_ptr |= 0x80000000;
        } else {
            self.child_ptr &= !0x80000000;
        }
    }

    fn get_ptr(&self) -> u32 {
        self.child_ptr & 0x7FFFFFFF
    }

    fn set_ptr(&mut self, ptr: u32) {
        let is_leaf = self.is_leaf();
        self.child_ptr = ptr & 0x7FFFFFFF;
        self.set_leaf(is_leaf);
    }
}

pub struct Sparse64Tree {
    pub nodes: Vec<Node64>, // Flat array of all nodes
}

impl Sparse64Tree {

     /// Optimized insert function avoiding Vec::insert and pointer updates
     pub fn bad_insert(&mut self, x: u32, y: u32, z: u32, depth: usize, color: [u8; 3]) {
        let mut current_index = 0;
        let mut current_depth = 0;

        while current_depth < depth {
            let child_index = self.compute_child_index(x, y, z, current_depth, depth);

            let needs_new_node = {
                let node = &self.nodes[current_index];
                (node.child_mask & (1 << child_index)) == 0
            };

            if needs_new_node {
                // Create new node at the end of the vector
                let new_node_index = self.nodes.len();
                self.nodes.push(Node64::default());

                // Update parent to point to the new child (using index)
                let parent = &mut self.nodes[current_index];
                if parent.child_mask == 0 {
                    parent.set_ptr(new_node_index as u32); // Set pointer to the first child block
                }
                parent.child_mask |= 1 << child_index;
                current_index = new_node_index;
            } else {
                // Navigate to existing child
                let node = &self.nodes[current_index];
                let parent_ptr = node.get_ptr();
                let rank = (node.child_mask & ((1 << child_index) - 1)).count_ones() as usize;
                current_index = parent_ptr as usize + rank;
            }

            current_depth += 1;
        }

        // Set leaf node data
        let leaf = &mut self.nodes[current_index];
        leaf.set_leaf(true);
        leaf.voxel_data = color;
    }

    /// Inserts a voxel at the given (x, y, z) and depth with the provided color.
    /// (Depth 0 would be the root; deeper levels refine the tree.)
    pub fn insert(&mut self, x: u32, y: u32, z: u32, depth: usize, color: [u8; 3]) {
        let mut current_index = 0;
        let mut current_depth = 0;

        while current_depth < depth {
            let child_index = self.compute_child_index(x, y, z, current_depth, depth);
            
            // Get node info without holding a reference
            let (needs_new_node, parent_mask, parent_ptr) = {
                let node = &self.nodes[current_index];
                (
                    (node.child_mask & (1 << child_index)) == 0,
                    node.child_mask,
                    node.get_ptr(),
                )
            };

            if needs_new_node {
                let insertion_index = if parent_mask == 0 {
                    // No children yet - append to end
                    let new_index = self.nodes.len();
                    let parent = &mut self.nodes[current_index];
                    parent.set_ptr(new_index as u32);
                    new_index
                } else {
                    // Insert into existing children block
                    let rank = (parent_mask & ((1 << child_index) - 1)).count_ones() as usize;
                    parent_ptr as usize + rank
                };

                // Create and insert new node
                let new_node = Node64 {
                    child_mask: 0,
                    child_ptr: 0,
                    voxel_data: [0, 0, 0],
                };
                self.nodes.insert(insertion_index, new_node);

                // Update parent's mask
                let parent = &mut self.nodes[current_index];
                parent.child_mask |= 1 << child_index;

                // Update all child pointers that need shifting
                for (i, node) in self.nodes.iter_mut().enumerate() {
                    if i != current_index {  // Skip the parent we just modified
                        let ptr = node.get_ptr();
                        if (ptr as usize) >= insertion_index {
                            node.set_ptr(ptr + 1);
                        }
                    }
                }

                current_index = insertion_index;
            } else {
                // Navigate to existing child
                let rank = (parent_mask & ((1 << child_index) - 1)).count_ones() as usize;
                current_index = parent_ptr as usize + rank;
            }

            current_depth += 1;
        }

        // Set leaf node data
        let leaf = &mut self.nodes[current_index];
        leaf.set_leaf(true);
        leaf.voxel_data = color;
    }

    pub fn old_insert(&mut self, x: u32, y: u32, z: u32, depth: usize, color: [u8; 3]) {
        let mut current_index = 0;
        let mut current_depth = 0;
    
        while current_depth < depth {
            let child_index = self.compute_child_index(x, y, z, current_depth, depth);
            
            let needs_new_node = {
                let node = &self.nodes[current_index];
                (node.child_mask & (1 << child_index)) == 0
            };
    
            if needs_new_node {
                // Create new node
                let new_node_index = self.nodes.len() as u32;
                self.nodes.push(Node64 {
                    child_mask: 0,
                    child_ptr: 0,
                    voxel_data: [0, 0, 0],
                });
    
                // Update parent
                let node = &mut self.nodes[current_index];
                if node.child_mask == 0 {
                    node.child_ptr = new_node_index;
                }
                node.child_mask |= 1 << child_index;
                current_index = new_node_index as usize;
            } else {
                // Navigate to existing child
                let node = &self.nodes[current_index];
                current_index = node.child_ptr as usize + child_index;
            }
    
            current_depth += 1;
        }
    
        // Set voxel data at the leaf node
        let leaf = &mut self.nodes[current_index];
        leaf.child_ptr |= 0x80000000; // Set leaf flag
        leaf.voxel_data = color;
    }


    /// Update all stored pointers in nodes that point to indices at or after insertion_index.
    /// We exclude the node at `exclude_index` (the parent we just modified) to leave its pointer intact.
    fn update_child_ptrs(&mut self, insertion_index: usize, exclude_index: usize) {
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if i == exclude_index {
                continue; // Skip the parent we just set up.
            }
            let flag = node.child_ptr & 0x8000_0000;
            let ptr = node.child_ptr & 0x7FFF_FFFF;
            // If this pointer refers to an index *after* the insertion point, update it.
            if (ptr as usize) >= insertion_index {
                node.child_ptr = flag | (ptr + 1);
            }
        }
    }
    
    fn compute_child_index(&self, x: u32, y: u32, z: u32, depth: usize, max_depth: usize) -> usize {
        let shift = (max_depth - depth - 1) * 2; // Reverse the shift order
        let x_part = ((x >> shift) & 0b11) as usize;
        let y_part = ((y >> shift) & 0b11) as usize;
        let z_part = ((z >> shift) & 0b11) as usize;
        z_part << 4 | y_part << 2 | x_part
    }

    fn broken_compute_child_index(&self, x: u32, y: u32, z: u32, depth: usize) -> usize {
        let shift = depth * 2; // 4 splits = 2 bits per level
        let x_part = ((x >> shift) & 0b11) as usize;
        let y_part = ((y >> shift) & 0b11) as usize;
        let z_part = ((z >> shift) & 0b11) as usize;
        z_part << 4 | y_part << 2 | x_part
    }

    pub fn flatten(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        for node in &self.nodes {
            buffer.extend_from_slice(&node.child_mask.to_le_bytes());
            buffer.extend_from_slice(&node.child_ptr.to_le_bytes());
            buffer.extend_from_slice(&node.voxel_data);
        }
        buffer
    }
}






use std::path::absolute;

///
/// 
///
/// 
/// 
/// 
/// 
///  GPU side nodes
/// 
/// 
/// 
/// 
/// 
/// 


use bytemuck::{Pod, Zeroable};
use cgmath::{abs_diff_eq, num_traits::pow};
use wgpu::{util::DeviceExt, BindGroup, BindGroupLayout, Buffer};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuNode64 {
    child_mask_low: u32,
    child_mask_high: u32,
    child_ptr_and_leaf: u32,
    _padding: u32,
    color: [f32; 3],
    _padding2: u32, // Keep 16-byte alignment
}

pub struct Tree64GpuManager {
    node_buffer: wgpu::Buffer,
    num_nodes: u32,
    pub contree_bind_group: BindGroup,
    pub contree_bind_group_layout: BindGroupLayout,
}

impl Tree64GpuManager {
    pub fn new(device: &wgpu::Device, contree: &Sparse64Tree) -> Self {
        let node_buffer = collect_nodes(contree, device);

        let contree_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // Binding for contree.
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

        // Create bind group.
        let contree_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &contree_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: node_buffer.as_entire_binding(),
                },
            ],
            label: Some("Contree Bind Group"),
        });


        Self {
            node_buffer,
            num_nodes: contree.nodes.len() as u32,
            contree_bind_group,
            contree_bind_group_layout,
        }
    }

    pub fn collect_nodes(&mut self, tree: &Sparse64Tree, device: &wgpu::Device) -> Buffer{
        let gpu_nodes: Vec<GpuNode64> = tree.nodes
            .iter()
            .map(|node| convert_node_to_gpu(node))
            .collect();

        let node_slice = bytemuck::cast_slice(&gpu_nodes);

        let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Indices Buffer"),
            contents: node_slice,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        return node_buffer;

    }

    // Upload tree to GPU
    pub fn upload_tree(&mut self, queue: &wgpu::Queue, tree: &Sparse64Tree) {
        let gpu_nodes: Vec<GpuNode64> = tree.nodes
            .iter()
            .map(|node| convert_node_to_gpu(node))
            .collect();

        self.num_nodes = gpu_nodes.len() as u32;

        queue.write_buffer(
            &self.node_buffer,
            0,
            bytemuck::cast_slice(&gpu_nodes),
        );
    }

    // Get buffer for binding
    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.node_buffer
    }
}


use noise::{Fbm, NoiseFn, Perlin};
// Test function to create a simple tree
pub fn create_test_tree() -> Sparse64Tree {
    let perlin = Perlin::new(1);

    let mut tree = Sparse64Tree {
        nodes: Vec::new(),
    };

    // Add root node
    tree.nodes.push(Node64 {
        child_mask: 0,
        child_ptr: 0,
        voxel_data: [0, 0, 0],
    });

    // Add some test voxels
    // let test_positions = [
    //     (0, 0, 0, [255, 0, 0]),   // Red voxel
    //     (2, 2, 2, [255, 255, 0]), // Yellow voxel
    //     (1, 1, 1, [0, 0, 255]),   // Blue voxel
    //     (3, 3, 3, [0, 255, 0]),   // Green voxel
    //     (3, 2, 1, [0, 0, 0]),   // Black voxel
    //     (2, 2, 1, [255, 255, 255]),   // WHITE voxel
    //     (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [0, 0, 255]),   // Blue voxel
    //     (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [0, 0, 255]),   // Blue voxel
    //     (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [0, 255, 255]),   // Blue voxel
    //     (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]),
    //     (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]),
    //     (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]),
    //     (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]),
    //     (generate_0_to_3(), generate_0_to_3(), generate_0_to_3(), [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]),
        
        
    // ];
    // let grid_size = 64;
    // for x in 0..grid_size {
    //     for y in 0..grid_size {
    //         for z in 0..grid_size {
    //             //let (num, greater) = generate_and_check();
    //             let val = perlin.get([(x as f64) / grid_size as f64, 0.0, (z as f64) / grid_size as f64]);

    //                 //tree.insert(x, y, z, 2, color);
    //             //println!("{}", val);
    //             if val.abs() > 0.0{
    //             //let num = x * (4 * 4) + y * 4 + z;
    //                 if val.abs() > 0.2 {
    //                     tree.insert(x, y, z, 3, [0, 121 + (generate_0_to_255() / 5), 40]);    
    //                 } else {
    //                     tree.insert(x, y, z, 3, [121 + (generate_0_to_255() / 15), 60, 0]);
    //                 }
                    
    //                 //tree.insert(x, y, z, 2, [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]);
    //                 //tree.insert(x, y, z, 1, [num as u8, num as u8, num as u8]);
    //             }
    //         }
    //     }
    // }

    let depth = 3;
    let past_grid: i32 = pow(4, depth);
    let grid_size= past_grid as u32;
    let noise_size = 32;
    for x in 0..grid_size {
        for z in 0..grid_size {
            let val = ((perlin.get([(x as f64) / noise_size as f64, (z as f64) / noise_size as f64]) * 10.0) + 10.0).abs() as u32;
            
            //let val = generate_0_to_3();
            for y in 0..val{
                //let (num, greater) = generate_and_check();
                

                    //tree.insert(x, y, z, 2, color);
                //println!("{}", val);
                
                //let num = x * (4 * 4) + y * 4 + z;
                    if y > 4 {
                        tree.insert(x, y, z, depth as usize, [0, 121 + (generate_0_to_255() / 5), 40]);    
                    } else {
                        tree.insert(x, y, z, depth as usize, [121 + (generate_0_to_255() / 15), 60, 0]);
                    }
                    
                    //tree.insert(x, y, z, 2, [generate_0_to_255(), generate_0_to_255(), generate_0_to_255()]);
                    //tree.insert(x, y, z, 1, [num as u8, num as u8, num as u8]);
                
            }
        }
    }



    // for (x, y, z, color) in test_positions.iter() {
    //     tree.insert(*x, *y, *z, 2, *color);
    // }

    // for node in &tree.nodes{
    //     println!("{:?}", node);
    // }
    
    tree
}


use rand::{random, Rng};

fn generate_0_to_3() -> u32 {
    let mut rng = rand::thread_rng();
    rng.gen_range(0..=6) // Inclusive range from 0 to 3
}

fn generate_0_to_255() -> u8 {
    let mut rng = rand::thread_rng();
    rng.gen_range(0..=255) as u8 // Inclusive range from 0 to 3
}
// Example usage:
/*
pub fn setup_tree_for_gpu(device: &wgpu::Device, queue: &wgpu::Queue) -> Tree64GpuManager {
    let mut gpu_manager = Tree64GpuManager::new(device);
    let test_tree = create_test_tree();
    gpu_manager.upload_tree(queue, &test_tree);
    gpu_manager
}
*/

// Convert CPU node to GPU format
fn convert_node_to_gpu(node: &Node64) -> GpuNode64 {

    //println!("Converting node - raw color: {:?}", node.voxel_data);
    
    let color = [
        node.voxel_data[0] as f32 / 255.0,
        node.voxel_data[1] as f32 / 255.0,
        node.voxel_data[2] as f32 / 255.0,
    ];

    //println!("Normalized color: {:?}", color);

    GpuNode64 {
        child_mask_low: (node.child_mask & 0xFFFFFFFF) as u32,
        child_mask_high: (node.child_mask >> 32) as u32,
        child_ptr_and_leaf: node.child_ptr | if node.is_leaf() { 0x80000000 } else { 0 },
        _padding: 0,
        color,
        _padding2: 0,
    }
}

pub fn collect_nodes(tree: &Sparse64Tree, device: &wgpu::Device) -> Buffer{
    let gpu_nodes: Vec<GpuNode64> = tree.nodes
        .iter()
        .map(|node| convert_node_to_gpu(node))
        .collect();

    let node_slice = bytemuck::cast_slice(&gpu_nodes);

    let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Indices Buffer"),
        contents: node_slice,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    return node_buffer;
}

fn generate_and_check() -> (u32, bool) {
    // Create a random number generator.  Using `thread_rng()` is usually sufficient.
    let mut rng = rand::thread_rng();

    // Generate a random number between 1 and 100 (inclusive).
    let number = rng.gen_range(1..=100); // Note the ..= for inclusive range

    // Check if the number is greater than 50.
    let is_greater_than_50 = number > 50;

    // Return the number and the boolean result as a tuple.
    (number, is_greater_than_50)
}