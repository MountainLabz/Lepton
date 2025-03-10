///
// Voxel Structs
///

const VOXEL_SCALE: f32 = 32.0;
const MAX_STEPS= 128u;
const EPSILON: f32 = 0.001;
const MAX_STACK_SIZE: i32 = 16;
const STEP_LIMIT = 256;

struct Node64 {
    child_mask_low: u32,
    child_mask_high: u32,
    child_ptr_and_leaf: u32,
    _padding: u32,
    color: vec3<f32>,
    _padding2: u32,
}

@group(2) @binding(0)
var<storage, read> nodes: array<Node64>;

///
// Tracing Structs
///

struct TraversalState {
    node: Node64,
    pos: vec3<f32>,
    depth: i32,
    current_voxel: vec3<i32>,
    direction: vec3<f32>,
    scale: f32,
}

struct RayHit {
    hit: bool,
    position: vec3<f32>,
    normal: vec3<f32>,
    color: vec3<f32>,
    distance: f32,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct ChildNodeRef {
    node: Node64,
    valid: bool,
}

struct StackState {
    node: Node64,
    node_position: vec3<f32>,
}

///
///
/// 64Tree Tracing stuff:
///
///

// Computes the DDA stepping parameters given a starting position, end position, and cell size.
// fn computeDDAState(start_pos: vec3<f32>, end_pos: vec3<f32>, voxel_size: vec3<f32>) -> TraversalState {
//     var state: TraversalState;
//     state.start_pos = start_pos;
//     state.end_pos = end_pos;
//     state.voxel_size = voxel_size;
//     state.direction = normalize(end_pos - start_pos);
//     state.current_voxel = vec3<i32>(floor(start_pos / voxel_size));
//     state.step = vec3<i32>(
//         select(-1, 1, state.direction.x > 0.0),
//         select(-1, 1, state.direction.y > 0.0),
//         select(-1, 1, state.direction.z > 0.0)
//     );
//     state.tMax = vec3<f32>(
//         // When moving in the x direction, if the ray is going positive then
//         // the next boundary is at (floor(x/size)+1)*size; otherwise at floor(x/size)*size.
//         ((select(floor(start_pos.x / voxel_size.x) + 1.0,
//                  floor(start_pos.x / voxel_size.x),
//                  state.direction.x < 0.0) * voxel_size.x) - start_pos.x) / state.direction.x,
//         ((select(floor(start_pos.y / voxel_size.y) + 1.0,
//                  floor(start_pos.y / voxel_size.y),
//                  state.direction.y < 0.0) * voxel_size.y) - start_pos.y) / state.direction.y,
//         ((select(floor(start_pos.z / voxel_size.z) + 1.0,
//                  floor(start_pos.z / voxel_size.z),
//                  state.direction.z < 0.0) * voxel_size.z) - start_pos.z) / state.direction.z
//     );
//     state.tDelta = vec3<f32>(
//         abs(voxel_size.x / state.direction.x),
//         abs(voxel_size.y / state.direction.y),
//         abs(voxel_size.z / state.direction.z)
//     );
//     // Also save the current node pointer (to be filled in later).
//     return state;
// }

fn has_children(node: Node64) -> bool {
    return (node.child_mask_low | node.child_mask_high) != 0u;
}

// Helper functions for bit manipulation
fn get_bit(mask: u32, index: u32) -> bool {
    return (mask & (1u << index)) != 0u;
}

fn bool_to_u32(b: bool) -> u32 {
    return select(0u, 1u, b);
}

fn get_child_index(pos: vec3<f32>, node_size: f32) -> u32 {
    let mid = vec3<u32>(
        bool_to_u32(pos.x >= 0.0),
        bool_to_u32(pos.y >= 0.0),
        bool_to_u32(pos.z >= 0.0)
    );
    return mid.x | (mid.y << 1u) | (mid.z << 2u);
}

fn get_child_mask(node: Node64, index: u32) -> bool {
    if (index < 32u) {
        return get_bit(node.child_mask_low, index);
    }
    return get_bit(node.child_mask_high, index - 32u);
}

// Get the next child pointer for traversal
fn get_child_ptr(node: Node64) -> u32 {
    return node.child_ptr_and_leaf & 0x7FFFFFFFu;
}

fn is_leaf(node: Node64) -> bool {
    return (node.child_ptr_and_leaf & 0x80000000u) != 0u;
}

fn count_set_bits_before(mask_low: u32, mask_high: u32, target_index: u32) -> u32 {
    var count: u32 = 0u;    
    // Check which mask we need to process
    if (target_index < 32u) {
        // Only check the low mask up to target_index
        for (var i: u32 = 0u; i < target_index; i++) {
            if ((mask_low & (1u << i)) != 0u) {
                count += 1u;
            }
        }
    } else {
        // Check all of mask_low and part of mask_high
        count = countOneBits(mask_low);  // Built-in popcount
        
        // Subtract 1 because we start counting from index 32
        for (var i: u32 = 0u; i < (target_index - 32u); i++) {
            if ((mask_high & (1u << i)) != 0u) {
                count += 1u;
            }
        }
    }
    return count;
}

fn sparse_get_child_at_coord(node: Node64, coord: vec3<i32>) -> ChildNodeRef {
    if (is_leaf(node)) {
        return ChildNodeRef(node, false);
    }

    let x = u32(clamp(coord.x, 0, 3));
    let y = u32(clamp(coord.y, 0, 3)); 
    let z = u32(clamp(coord.z, 0, 3));
    let target_idx = x + (y * 4u) + (z * 16u);

    if (!get_child_mask(node, target_idx)) {
        return ChildNodeRef(node, false);
    }

    // Calculate number of children before this one
    let count = count_set_bits_before(node.child_mask_low, node.child_mask_high, target_idx);
    
    // Get child pointer offset
    let child_ptr = get_child_ptr(node);
    return ChildNodeRef(nodes[child_ptr + count], true);
}

// Main ray-box intersection function
fn ray_box_intersection(ray: Ray, box_min: vec3<f32>, box_size: f32) -> vec2<f32> {
    let box_max = box_min + vec3<f32>(box_size);
    
    let t1 = (box_min - ray.origin) / ray.direction;
    let t2 = (box_max - ray.origin) / ray.direction;
    
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    
    let enter = max(max(tmin.x, tmin.y), tmin.z);
    let exit = min(min(tmax.x, tmax.y), tmax.z);
    
    return vec2<f32>(enter, exit);
}

fn calculate_normal(pos: vec3<f32>, center: vec3<f32>) -> vec3<f32> {
    let delta = pos - center;
    let abs_delta = abs(delta);
    let max_comp = max(max(abs_delta.x, abs_delta.y), abs_delta.z);
    
    if (max_comp == abs_delta.x) {
        return vec3<f32>(sign(delta.x), 0.0, 0.0);
    } else if (max_comp == abs_delta.y) {
        return vec3<f32>(0.0, sign(delta.y), 0.0);
    }
    return vec3<f32>(0.0, 0.0, sign(delta.z));
}

fn within_bounds(position: vec3<i32>, array_size: vec3<i32>) -> bool{

    if (position.x > array_size.x || position.x < -1){
        return false;
    }
    if (position.y > array_size.y || position.y < -1 ){
        return false;
    }
    if (position.z > array_size.z || position.z < -1){
        return false;
    }

   return true;
}

fn is_filled(node: Node64, coord: vec3<i32>) -> bool {
    // Clamp coordinates to valid range
    let x = u32(clamp(coord.x, 0, 3));
    let y = u32(clamp(coord.y, 0, 3));
    let z = u32(clamp(coord.z, 0, 3));

    // Compute the bit index for the given coordinate
    let index = x + (y * 4u) + (z * 16u);

    // Check if the bit is set in the child mask
    return get_child_mask(node, index);
}



// Given a position in world (or node) space and the current node's scale (cell size),
// this function computes the local grid coordinate (in the range [0, 3]) within the node.
fn get_grid_coord(pos: vec3<f32>, scale: f32) -> vec3<i32> {
    // Assume node boundaries are aligned to multiples of 'scale'.
    // Compute the lower corner of the node.
    let cell_min = floor(pos / scale) * scale;
    
    // Get the local position within the node.
    let local = pos - cell_min;
    
    // Each child cell is scale/4 in size.
    let child_cell_size = scale / 4.0;
    
    // Compute the grid coordinate by dividing the local coordinate by the child cell size.
    // floor() makes sure we get an integer value in [0, 3] (if pos is within the node).
    let coord = vec3<i32>(floor(local / child_cell_size));
    
    return coord;
}



///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

fn old_intersect_planes(origin: vec3<f32>, inv_dir: vec3<f32>, cell_min: vec3<f32>, cell_size: vec3<f32>) -> f32 {
    let side_pos = cell_min + step(vec3<f32>(0.0), inv_dir) * cell_size;
    let side_dist = (side_pos - origin) * inv_dir;
    
    return min(min(side_dist.x, side_dist.y), side_dist.z) + EPSILON;
}

// Adapted intersect_planes: returns the new distance along the ray.
fn new_old_intersect_planes(origin: vec3<f32>, dir: vec3<f32>, cell_min: vec3<f32>, scale: f32) -> f32 {
    // Compute the position of the "far" face on each axis.
    let side_pos = cell_min + step(vec3<f32>(0.0), dir) * scale;
    // Calculate the distance t to each face (assuming dir is nonzero).
    let side_dist = (side_pos - origin) / dir;
    let tmax = min(min(side_dist.x, side_dist.y), side_dist.z);

    // Compute the neighboring cell's bounds.
    var neighborMin = cell_min;
    if (tmax == side_dist.x) {
        neighborMin.x = cell_min.x + copysign(scale, dir.x);
    }
    if (tmax == side_dist.y) {
        neighborMin.y = cell_min.y + copysign(scale, dir.y);
    }
    if (tmax == side_dist.z) {
        neighborMin.z = cell_min.z + copysign(scale, dir.z);
    }
    let neighborMax = neighborMin + vec3<f32>(scale);

    // Compute the intersection position along the ray.
    let pos = origin + dir * tmax;
    // Clamp the position to ensure it's inside the neighbor's bounds.
    let clampedPos = clamp(pos, neighborMin, neighborMax);

    // If your ray direction is normalized, the distance is simply the length of the displacement.
    return length(clampedPos - origin);
}

fn intersect_planes(origin: vec3<f32>, direction: vec3<f32>, cell_min: vec3<f32>, cell_max: vec3<f32>) -> f32 {
    var t_min: f32 = 1e30; // start with a very large value
    let eps: f32 = 1e-6;   // small epsilon to avoid division by zero

    // X-axis
    if (abs(direction.x) > eps) {
        var t_x: f32;
        if (direction.x > 0.0) {
            t_x = (cell_max.x - origin.x) / direction.x;
        } else {
            t_x = (cell_min.x - origin.x) / direction.x;
        }
        t_min = min(t_min, t_x);
    }

    // Y-axis
    if (abs(direction.y) > eps) {
        var t_y: f32;
        if (direction.y > 0.0) {
            t_y = (cell_max.y - origin.y) / direction.y;
        } else {
            t_y = (cell_min.y - origin.y) / direction.y;
        }
        t_min = min(t_min, t_y);
    }

    // Z-axis
    if (abs(direction.z) > eps) {
        var t_z: f32;
        if (direction.z > 0.0) {
            t_z = (cell_max.z - origin.z) / direction.z;
        } else {
            t_z = (cell_min.z - origin.z) / direction.z;
        }
        t_min = min(t_min, t_z);
    }

    return t_min;
}

fn floor_scale(pos: vec3<f32>, scale: f32) -> vec3<f32> {
    return floor(pos / scale) * scale;
}

fn get_node_cell_index(pos: vec3<f32>, scale_exp: i32) -> u32 {
    let cell_pos = (bitcast<vec3<u32>>(pos) >> vec3<u32>(u32(scale_exp))) & vec3<u32>(3u);
    return cell_pos.x + cell_pos.z * 4u + cell_pos.y * 16u;
}

fn get_node_cell_index_nb(node: Node64, pos: vec3<f32>, unaltered_scale: f32) -> u32 {
    let scale = unaltered_scale * 4.0;
    // First, find the lower corner (cell_min) of the node that contains 'pos'.
    // This assumes that node boundaries are aligned to multiples of 'scale'.
    let cell_min = floor(pos / scale) * scale;
    
    // Compute the local coordinate within the node.
    // 'local' will be in the range [0, scale) for each component.
    let local = pos - cell_min;
    
    // Each node is subdivided into 4 cells along each axis.
    // Therefore, the size of each child cell is scale / 4.
    // Compute the child cell index along each axis by finding where
    // the local coordinate falls in that subdivision.
    let cell_x = u32(floor(local.x * 4.0 / scale));
    let cell_y = u32(floor(local.y * 4.0 / scale));
    let cell_z = u32(floor(local.z * 4.0 / scale));

    
    
    // Combine the per-axis indices into a single index.
    // The ordering is: index = cell_x + cell_z * 4 + cell_y * 16.
    return (cell_x + cell_z * 4u + cell_y * 16u) + get_child_ptr(node) + 1u;
}

fn old_true_trace(ray: Ray, tree_pos: vec3<f32>, root_size: f32) -> RayHit{
    var steps = 0u;

    var result: RayHit;
    result.color = vec3<f32>(0.0);
    result.hit = false;
    result.distance = 99999.0;

    var node_stack:array<Node64, MAX_STACK_SIZE>;
    var node_ptr: u32;
    var ray_position: vec3<f32>;
    var parent_node: Node64;
    var child_node: Node64;
    var depth: i32;


    //init

    let root_intersection = ray_box_intersection(ray, tree_pos, root_size);
    let root_start = ray.origin + ray.direction * root_intersection.x;
    let root_end = ray.origin + ray.direction * root_intersection.y;
    if (root_intersection.y < root_intersection.x || root_intersection.y < 0.0) {
        return result;
    }

   

    result.color = vec3<f32>(-0.2);
    if (!has_children(nodes[0])) {
        return result;
    }
    
    // now that the intersection checks are complete we init the cycle.
    ray_position = root_start + (ray.direction * 0.001);
    parent_node = nodes[0];
    node_stack[0] = parent_node;
    node_ptr = 0u;
    depth = 1;

    
    while(all(ray_position >= tree_pos) && all(ray_position <= tree_pos + vec3<f32>(root_size))){

        if (steps >= MAX_STEPS){
            result.hit = true;
            result.color = vec3<f32>(-1.0);
            return result;

        }

        var current_scale = root_size / pow(4.0, f32(depth));

        // get the current child node of the current parent node [SUSPICIOUS]
        var child_idx = get_node_cell_index_nb(parent_node, ray_position, current_scale);
        var child_node = nodes[child_idx];

        // decend until either a leaf or does not have children!
        while (!is_leaf(child_node) && has_children(child_node)) {
            // increase the depth and calc size
            depth++;
            current_scale = root_size / pow(4.0, f32(depth));

            // reset the new node
            parent_node = child_node;
            child_idx = get_node_cell_index_nb(parent_node, ray_position, current_scale);
            child_node = nodes[child_idx];
            steps++;

            node_ptr++;
            node_stack[node_ptr] = parent_node;

            result.color = vec3<f32>(f32(depth) / 4.0);

            if (steps >= MAX_STEPS){
                result.hit = true;
                result.color = vec3<f32>(-1.0);
                return result;
            }
        }

        // check for leaf
        if (is_leaf(child_node)){
            result.color = child_node.color;
            result.hit = true;
            return result;
        }
        // otherwise:
        // exit the current node
        let node_intersection = ray_box_intersection(Ray(ray_position, ray.direction), floor_scale(ray_position, current_scale), current_scale);
        let root_end = ray.origin + ray.direction * root_intersection.y;

        ray_position = root_end + (ray.direction * 0.001);
        //ray_position = ray_position + intersect_planes(ray_position, -ray.direction, floor_scale(ray_position, current_scale), vec3<f32>(current_scale));
        
        //check if we have exited the current parent node
        if ((!all(ray_position > vec3<f32>(current_scale * 4.0)) || !all(ray_position < vec3<f32>(floor_scale(ray_position, current_scale * 4.0))))){
            depth--;
            if (node_ptr >= 1u) { node_ptr -= 1u; } else { return result;}
            parent_node = node_stack[node_ptr];
        } else {
            depth -= 2;
            if (node_ptr >= 2u) { node_ptr -= 2u; } else { return result;}
            parent_node = node_stack[node_ptr];
        }
        
        if (node_ptr <= 0u){
            return result;
        }


    }



    return result;
}

fn return_scale(root_scale: f32, depth: i32) -> f32{
    return root_scale / pow(4.0, f32(depth));
}

fn is_point_in_aabb(origin: vec3<f32>, box_position: vec3<f32>, box_scale: vec3<f32>) -> bool {
    let box_max = box_position + box_scale;

    return origin.x >= box_position.x && origin.x <= box_max.x &&
           origin.y >= box_position.y && origin.y <= box_max.y &&
           origin.z >= box_position.z && origin.z <= box_max.z;
}

// helper for clamping.
fn copysign(x: f32, y: f32) -> f32 {
    // Returns x with the sign of y.
    if (y < 0.0) {
        return -abs(x);
    }
    return abs(x);
}


fn compute_side_dist(ray_origin: vec3<f32>, ray_direction: vec3<f32>, voxel_size: vec3<f32>) -> vec3<f32> {
    let voxel_pos = floor(ray_origin / voxel_size) * voxel_size; // Get the lower bound of the voxel

    // Compute the next boundary per axis
    let next_boundary = voxel_pos + step(vec3<f32>(0.0), ray_direction) * voxel_size;

    // Compute side distances
    let side_dist = (next_boundary - ray_origin) / ray_direction;

    return side_dist;
}

fn true_trace(ray: Ray, tree_pos: vec3<f32>, root_size: f32) -> RayHit {
    var deepest_depth = 0;

    var result: RayHit;
    result.color = vec3<f32>(0.0);
    result.hit = false;
    result.distance = 99999.0;

    var root_start: vec3<f32>;

    if (is_point_in_aabb(ray.origin, tree_pos, vec3<f32>(root_size))){
        root_start = ray.origin;
    } else {
        let root_intersection = ray_box_intersection(ray, tree_pos, root_size);
        root_start = ray.origin + ray.direction * root_intersection.x;
        let root_end = ray.origin + ray.direction * root_intersection.y;
        if (root_intersection.y < root_intersection.x || root_intersection.y < 0.0) {
            return result;
        }
        result.color = vec3<f32>(0.0);
    }

    // init vars
    var root_node = nodes[0];
    var ray_position = root_start + (ray.direction * EPSILON); // the ray position in global space
    var depth = 1;
    var parent_size: f32 = return_scale(VOXEL_SCALE, 0);
    var child_size: f32 = return_scale(VOXEL_SCALE, depth);
    var current_voxel: vec3<i32> = vec3<i32>(floor((ray_position - floor_scale(ray_position, parent_size)) / child_size)); // the position in vox coords in the current node
    var child_node: Node64;
    var parent_node = root_node;
        // stack init
    var step = 0; // current step
    var stack: array<Node64, MAX_STACK_SIZE>; // stack of previous nodes
    var stack_ptr: u32 = 0u; // stack pointer
    stack[stack_ptr] = root_node; // setting starting node to the root node

    // traversal loop
    while (step < STEP_LIMIT){ // we check if we are inside our step limit
        if(depth > deepest_depth){
            deepest_depth = depth;
        }

        if (!all(ray_position >= tree_pos) || !all(ray_position <= tree_pos + vec3<f32>(root_size))){ // if we leave the root node return far
            result.color = vec3<f32>(-0.1 * f32(deepest_depth));
            return result;
        }
        var current_voxel: vec3<i32> = vec3<i32>(floor((ray_position - floor_scale(ray_position, parent_size)) / child_size)); // the position in vox coords in the current node
        let node_reference_ref = sparse_get_child_at_coord(parent_node, current_voxel);
        if (node_reference_ref.valid == true){
            let node_reference = node_reference_ref.node;
            if is_leaf(node_reference) {
                result.color = node_reference.color;
                result.hit = true;
                return result;
            } else if (has_children(node_reference)) {
                // decend into the node by:
                // continuing the stack!
                stack_ptr = stack_ptr + 1u;
                stack[stack_ptr] = node_reference;
                // setting the new "parent" node
                parent_node = node_reference;
                //increasing depth
                depth++;
                // setting the scales
                parent_size = return_scale(VOXEL_SCALE, depth - 1);
                child_size = return_scale(VOXEL_SCALE, depth);

                step++;
                continue;
            }
        }
            // if neither of those, then keep marching!
            let last_pos = ray_position;
            
            let distance = old_intersect_planes(ray_position, ray.direction, floor_scale(ray_position, parent_size), floor_scale(ray_position, parent_size) + parent_size);
            ray_position = ray_position + (ray.direction * distance);

            if (!all(ray_position >= floor_scale(last_pos, parent_size)) || !all(ray_position <= floor_scale(last_pos, parent_size) + parent_size)){
                // recend into the node by:
                // backing up the stack
                stack_ptr = stack_ptr - 1u;
                // getting the old parent node
                parent_node = stack[stack_ptr];
                //decreasing depth
                depth--;
                // reversing the scales
                parent_size = return_scale(VOXEL_SCALE, depth - 1);
                child_size = return_scale(VOXEL_SCALE, depth);

                step++;
                continue;
            }



        



        step++;
    }

    //result.color = vec3<f32>(-1.0);

    return result;
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


fn clamp_trace(ray: Ray, tree_pos: vec3<f32>, root_size: f32) -> RayHit {
    var deepest_depth = 0;

    var result: RayHit;
    result.color = vec3<f32>(0.0);
    result.hit = false;
    result.distance = 99999.0;

    var root_start: vec3<f32>;

    if (is_point_in_aabb(ray.origin, tree_pos, vec3<f32>(root_size))){
        root_start = ray.origin;
    } else {
        let root_intersection = ray_box_intersection(ray, tree_pos, root_size);
        root_start = ray.origin + ray.direction * root_intersection.x;
        let root_end = ray.origin + ray.direction * root_intersection.y;
        if (root_intersection.y < root_intersection.x || root_intersection.y < 0.0) {
            return result;
        }
        result.color = vec3<f32>(0.0);
    }

    // init vars
    var root_node = nodes[0];
    var ray_position = root_start + (ray.direction * EPSILON); // the ray position in global space
    var depth = 1;
    var parent_size: f32 = return_scale(VOXEL_SCALE, 0);
    var child_size: f32 = return_scale(VOXEL_SCALE, depth);
    var current_voxel: vec3<i32> = vec3<i32>(floor((ray_position - floor_scale(ray_position, parent_size)) / child_size)); // the position in vox coords in the current node
    var child_node: Node64;
    var parent_node = root_node;
    var cell_origin = floor_scale(ray_position, parent_size);
        // stack init
    var step = 0; // current step
    var stack: array<Node64, MAX_STACK_SIZE>; // stack of previous nodes
    var stack_ptr: u32 = 0u; // stack pointer
    stack[stack_ptr] = root_node; // setting starting node to the root node

    // traversal loop
    while (step < STEP_LIMIT){ // we check if we are inside our step limit

        // debug for node depth vision
        if(depth > deepest_depth){
            deepest_depth = depth;
        }

        if (any(ray_position < tree_pos) || any(ray_position > tree_pos + VOXEL_SCALE)){ // if we leave the root node return far
            result.color = vec3<f32>(-0.2 * f32(deepest_depth));
            return result;
        }
        //var current_voxel: vec3<i32> = vec3<i32>(floor((ray_position - floor_scale(ray_position, parent_size)) / child_size)); // the position in vox coords in the current node
        current_voxel = vec3<i32>(floor((ray_position - cell_origin) / child_size));
        
        let node_reference_ref = sparse_get_child_at_coord(parent_node, current_voxel);

        //     Conditions for Node Validity
        // A child node is considered valid when all of the following conditions are met:

        // The parent node is not a leaf node (checked with is_leaf(node))
        // The requested coordinates are mapped to a valid index within the 64-tree (4×4×4 grid)
        // The child mask at the target index is set (checked with get_child_mask(node, target_idx))
        // The child node exists at the calculated offset in the nodes array

        // In more detail:

        // First, the function checks if the input node is a leaf. If it is, it returns the node with valid=false since leaf nodes don't have children.
        // It clamps the coordinates to the range [0,3] for each axis, effectively handling the 4×4×4 grid structure.
        // It calculates the target index in the flattened array using the formula x + (y * 4) + (z * 16).
        // It checks if the bit at target_idx in the child mask is set. If not, returns with valid=false.
        // If the bit is set, it:

        // Counts how many bits are set before this index in the child mask
        // Gets the child pointer base index
        // Returns the node at the calculated offset with valid=true
        let node_reference = node_reference_ref.node;
        if (node_reference_ref.valid == true){
                       
            if (has_children(node_reference)) {
                // decend into the node by:
                // continuing the stack!
                stack_ptr = stack_ptr + 1u;
                stack[stack_ptr] = node_reference;
                // setting the new "parent" node
                parent_node = node_reference;
                //increasing depth
                depth++;
                // setting the scales
                parent_size = return_scale(VOXEL_SCALE, depth - 1);
                child_size = return_scale(VOXEL_SCALE, depth);
                cell_origin = floor_scale(ray_position, parent_size);

                step++;
                continue;
            }
            
            if is_leaf(node_reference) {
                result.color = node_reference.color;    
                
                result.hit = true;



                return result;

            }

        } 
            
            // if neither of those, then keep marching!
            let last_pos = ray_position;

            let dda_size = child_size;
            let side_dist= compute_side_dist(ray_position, ray.direction, vec3<f32>(dda_size));

            let tmax = min(min(side_dist.x, side_dist.y), side_dist.z);

            // Compute neighbor_min using the axis that caused the exit
            var neighbor_min = floor_scale(ray_position, dda_size);

            if (tmax == side_dist.x) {
                neighbor_min.x += sign(ray.direction.x) * dda_size;
            }
            if (tmax == side_dist.y) {
                neighbor_min.y += sign(ray.direction.y) * dda_size;
            }
            if (tmax == side_dist.z) {
                neighbor_min.z += sign(ray.direction.z) * dda_size;
            }

            // Compute neighbor_max
            let neighbor_max = neighbor_min + vec3<f32>(dda_size);

            // Clamp the ray position to ensure it stays within the correct voxel
            
            // ray_position = clamp(ray_position + ray.direction * tmax, neighbor_min, neighbor_max);
            // ray_position = ray_position + (ray.direction * 1.0);

            ray_position = last_pos + ray.direction * tmax;
            // Then nudge just enough to ensure you're inside the next voxel:
            ray_position = ray_position + (ray.direction * EPSILON);

            //let distance = old_intersect_planes(ray_position, ray.direction, floor_scale(ray_position, parent_size), floor_scale(ray_position, parent_size) + parent_size);
            //ray_position = ray_position + (ray.direction * distance);

            
            // if we have exited the parent node then continue
            if (!all(ray_position >= cell_origin) || !all(ray_position <= cell_origin + parent_size)){
                // recend into the node by:
                // backing up the stack
                if (stack_ptr == 0u) {
                    result.color = vec3<f32>(-0.1 * f32(deepest_depth));
                    return result;
                }

                stack_ptr = stack_ptr - 1u;
               
                // getting the old parent node
                parent_node = stack[stack_ptr];
                //decreasing depth
                depth--;
                // reversing the scales
                parent_size = return_scale(VOXEL_SCALE, depth - 1);
                child_size = return_scale(VOXEL_SCALE, depth);
                // Also update the cell_origin for the parent.
                cell_origin = floor_scale(ray_position, parent_size);

                step++;
                continue;
            }

            

            // // if we have exited the parent node then continue
            // if (!all(ray_position >= floor_scale(last_pos, parent_size)) || !all(last_pos <= floor_scale(ray_position, parent_size) + parent_size)){
            //     // recend into the node by:
            //     // backing up the stack
            //     stack_ptr = stack_ptr - 1u;
            //     // getting the old parent node
            //     parent_node = stack[stack_ptr];
            //     //decreasing depth
            //     depth--;
            //     // reversing the scales
            //     parent_size = return_scale(VOXEL_SCALE, depth - 1);
            //     child_size = return_scale(VOXEL_SCALE, depth);
            //     // Also update the cell_origin for the parent.
            //     cell_origin = floor_scale(ray_position, parent_size);

            //     step++;
            //     continue;
            // }
            

        step++;
    }

    return result;
}











fn clamp_trace2(ray: Ray, tree_pos: vec3<f32>, root_size: f32) -> RayHit {
    var deepest_depth = 0;

    var result: RayHit;
    result.color = vec3<f32>(0.0);
    result.hit = false;
    result.distance = 99999.0;

    var root_start: vec3<f32>;

    if (is_point_in_aabb(ray.origin, tree_pos, vec3<f32>(root_size))){
        root_start = ray.origin;
    } else {
        let root_intersection = ray_box_intersection(ray, tree_pos, root_size);
        root_start = ray.origin + ray.direction * root_intersection.x;
        let root_end = ray.origin + ray.direction * root_intersection.y;
        if (root_intersection.y < root_intersection.x || root_intersection.y < 0.0) {
            return result;
        }
        result.color = vec3<f32>(0.0);
    }

    // init vars
    var root_node = nodes[0];
    var ray_position = root_start + (ray.direction * EPSILON); // the ray position in global space
    var depth = 1;
    var parent_size: f32 = return_scale(VOXEL_SCALE, 0);
    var child_size: f32 = return_scale(VOXEL_SCALE, depth);
    var current_voxel: vec3<i32> = vec3<i32>(floor((ray_position - floor_scale(ray_position, parent_size)) / child_size)); // the position in vox coords in the current node
    var child_node: Node64;
    var parent_node = root_node;
    var cell_origin = floor_scale(ray_position, parent_size);
        // stack init
    var step = 0; // current step
    var stack: array<StackState, MAX_STACK_SIZE>; // stack of previous nodes
    var stack_ptr: u32 = 0u; // stack pointer

    let first_stack_state = StackState(root_node, cell_origin);
    stack[stack_ptr] = first_stack_state; // setting starting node to the root node

    // traversal loop
    while (step < STEP_LIMIT){ // we check if we are inside our step limit

        // debug for node depth vision
        if(depth > deepest_depth){
            deepest_depth = depth;
        }

        if (any(ray_position < tree_pos) || any(ray_position > tree_pos + VOXEL_SCALE)){ // if we leave the root node return far
            result.color = vec3<f32>(-0.2 * f32(deepest_depth));
            return result;
        }
        //var current_voxel: vec3<i32> = vec3<i32>(floor((ray_position - floor_scale(ray_position, parent_size)) / child_size)); // the position in vox coords in the current node
        current_voxel = vec3<i32>(floor((ray_position - cell_origin) / child_size));
        
        let node_reference_ref = sparse_get_child_at_coord(parent_node, current_voxel);

        //     Conditions for Node Validity
        // A child node is considered valid when all of the following conditions are met:

        // The parent node is not a leaf node (checked with is_leaf(node))
        // The requested coordinates are mapped to a valid index within the 64-tree (4×4×4 grid)
        // The child mask at the target index is set (checked with get_child_mask(node, target_idx))
        // The child node exists at the calculated offset in the nodes array

        // In more detail:

        // First, the function checks if the input node is a leaf. If it is, it returns the node with valid=false since leaf nodes don't have children.
        // It clamps the coordinates to the range [0,3] for each axis, effectively handling the 4×4×4 grid structure.
        // It calculates the target index in the flattened array using the formula x + (y * 4) + (z * 16).
        // It checks if the bit at target_idx in the child mask is set. If not, returns with valid=false.
        // If the bit is set, it:

        // Counts how many bits are set before this index in the child mask
        // Gets the child pointer base index
        // Returns the node at the calculated offset with valid=true
        let node_reference = node_reference_ref.node;
        if (node_reference_ref.valid == true){
                       
            if (has_children(node_reference)) {
                // decend into the node by:
                // continuing the stack!
                let child_voxel_origin = cell_origin + vec3<f32>(current_voxel) * child_size;
                // setting the new "parent" node
                parent_node = node_reference;
                //increasing depth
                depth++;
                // setting the scales
                parent_size = return_scale(VOXEL_SCALE, depth - 1);
                child_size = return_scale(VOXEL_SCALE, depth);



                //cell_origin = floor_scale(ray_position, parent_size);
                
                cell_origin = child_voxel_origin;




                let new_stack_state = StackState(node_reference, cell_origin);
                stack_ptr = stack_ptr + 1u;
                stack[stack_ptr] = new_stack_state;
                step++;
                continue;
            }
            
            if is_leaf(node_reference) {
                result.color = node_reference.color;    
                
                result.hit = true;



                return result;

            }

        } 
            
            // if neither of those, then keep marching!
            let last_pos = ray_position;

            let dda_size = child_size;
            let side_dist= compute_side_dist(ray_position, ray.direction, vec3<f32>(dda_size));

            let tmax = min(min(side_dist.x, side_dist.y), side_dist.z);

            // Compute neighbor_min using the axis that caused the exit
            var neighbor_min = floor_scale(ray_position, dda_size);

            if (tmax == side_dist.x) {
                neighbor_min.x += sign(ray.direction.x) * dda_size;
            }
            if (tmax == side_dist.y) {
                neighbor_min.y += sign(ray.direction.y) * dda_size;
            }
            if (tmax == side_dist.z) {
                neighbor_min.z += sign(ray.direction.z) * dda_size;
            }

            // Compute neighbor_max
            let neighbor_max = neighbor_min + vec3<f32>(dda_size);

            // Clamp the ray position to ensure it stays within the correct voxel
            
            // ray_position = clamp(ray_position + ray.direction * tmax, neighbor_min, neighbor_max);
            // ray_position = ray_position + (ray.direction * 1.0);

            ray_position = last_pos + ray.direction * tmax;
            // Then nudge just enough to ensure you're inside the next voxel:
            ray_position = ray_position + (ray.direction * EPSILON);

            //let distance = old_intersect_planes(ray_position, ray.direction, floor_scale(ray_position, parent_size), floor_scale(ray_position, parent_size) + parent_size);
            //ray_position = ray_position + (ray.direction * distance);

            
            // if we have exited the parent node then continue
            if (!all(ray_position >= cell_origin) || !all(ray_position <= cell_origin + parent_size)){
                // recend into the node by:
                // backing up the stack
                if (stack_ptr == 0u) {
                    result.color = vec3<f32>(-0.1 * f32(deepest_depth));
                    return result;
                }

                stack_ptr = stack_ptr - 1u;
               
                // getting the old parent node
                parent_node = stack[stack_ptr].node;
                //decreasing depth
                depth--;
                // reversing the scales
                parent_size = return_scale(VOXEL_SCALE, depth - 1);
                child_size = return_scale(VOXEL_SCALE, depth);
                // Also update the cell_origin for the parent.
                cell_origin = stack[stack_ptr].node_position;

                step++;
                continue;
            }

            

            // if we have exited the parent node then continue
            // if (!all(ray_position >= floor_scale(last_pos, parent_size)) || !all(last_pos <= floor_scale(ray_position, parent_size) + parent_size)){
            //     // recend into the node by:
            //     // backing up the stack
            //     stack_ptr = stack_ptr - 1u;
            //     // getting the old parent node
            //     parent_node = stack[stack_ptr].node;
            //     //decreasing depth
            //     depth--;
            //     // reversing the scales
            //     parent_size = return_scale(VOXEL_SCALE, depth - 1);
            //     child_size = return_scale(VOXEL_SCALE, depth);
            //     // Also update the cell_origin for the parent.
            //     cell_origin = stack[stack_ptr].node_position;

            //     step++;
            //     continue;
            // }
            

        step++;
    }

    return result;
}















fn trace_tree(ray: Ray, tree_pos: vec3<f32>, root_size: f32) -> RayHit {
    var steps: u32 = 0;

    var result: RayHit;
    result.color = vec3<f32>(0.0);
    result.hit = false;
    result.distance = 99999.0;

    var root_start: vec3<f32>;

    if (is_point_in_aabb(ray.origin, tree_pos, vec3<f32>(root_size))){
        root_start = ray.origin;
    } else {
        let root_intersection = ray_box_intersection(ray, tree_pos, root_size);
        root_start = ray.origin + ray.direction * root_intersection.x;
        let root_end = ray.origin + ray.direction * root_intersection.y;
        if (root_intersection.y < root_intersection.x || root_intersection.y < 0.0) {
            return result;
        }
        result.color = vec3<f32>(-0.2);
    }

    var ray_position = root_start + (ray.direction * EPSILON);
    var depth = 1;
    var parent_size = root_size;
    var child_size = return_scale(root_size, depth);
    var cell_origin = tree_pos;
    var current_voxel = vec3<i32>(floor((ray_position - cell_origin) / child_size));
    var current_node: Node64 = nodes[0];

    var stack: array<StackState, MAX_STACK_SIZE>;
    var stack_ptr = 0u;
    var stack_state = StackState(nodes[0], cell_origin);


    while (steps <= MAX_STEPS) {
        if (any(ray_position < tree_pos) || any(ray_position > tree_pos + VOXEL_SCALE)){ // if we leave the root node return far
            return result;
        }


        current_voxel = vec3<i32>(floor((ray_position - cell_origin) / child_size));
        let node_reference_ref = sparse_get_child_at_coord(current_node, current_voxel);

        if (node_reference_ref.valid == true) {
            if (has_children(node_reference_ref.node)){
                let child_voxel_origin = cell_origin + vec3<f32>(current_voxel) * child_size;
                depth++;
                
                current_node = node_reference_ref.node;
                parent_size = child_size;
                child_size = return_scale(VOXEL_SCALE, depth);
                cell_origin = child_voxel_origin;

                stack_state.node = node_reference_ref.node;
                stack_state.node_position = cell_origin;

                stack_ptr++;
                stack[stack_ptr] = stack_state;

                continue;
            }

            

        }

        if is_leaf(node_reference_ref.node) {
            result.color = node_reference_ref.node.color;    
                
            result.hit = true;

            return result;

        }

    // if neither of those, then keep marching!
            let last_pos = ray_position;

            let dda_size = child_size;
            let side_dist= compute_side_dist(ray_position, ray.direction, vec3<f32>(dda_size));

            let tmax = min(min(side_dist.x, side_dist.y), side_dist.z);

            // Compute neighbor_min using the axis that caused the exit
            var neighbor_min = floor_scale(ray_position, dda_size);

            if (tmax == side_dist.x) {
                neighbor_min.x += sign(ray.direction.x) * dda_size;
            }
            if (tmax == side_dist.y) {
                neighbor_min.y += sign(ray.direction.y) * dda_size;
            }
            if (tmax == side_dist.z) {
                neighbor_min.z += sign(ray.direction.z) * dda_size;
            }

            // Compute neighbor_max
            let neighbor_max = neighbor_min + vec3<f32>(dda_size);

            // Clamp the ray position to ensure it stays within the correct voxel
            
            // ray_position = clamp(ray_position + ray.direction * tmax, neighbor_min, neighbor_max);
            // ray_position = ray_position + (ray.direction * 1.0);

            ray_position = last_pos + ray.direction * tmax;
            // Then nudge just enough to ensure you're inside the next voxel:
            ray_position = ray_position + (ray.direction * EPSILON);


        //let distance = old_intersect_planes(ray_position, ray.direction, floor_scale(ray_position, parent_size), floor_scale(ray_position, parent_size) + parent_size);
        //ray_position = ray_position + (ray.direction * distance) + (ray.direction * EPSILON);


        // reset
        current_node = nodes[0];
        depth = 1;
        parent_size = root_size;
        child_size = return_scale(root_size, depth);
        cell_origin = tree_pos;

        steps++;
    }



    return result;
}







///
///
/// camera calculations
///
///

struct CameraOrientation {
    cameraFront: vec3<f32>,
    cameraRight: vec3<f32>,
    cameraUp: vec3<f32>,
};

fn flatten_index(x: i32, y: i32, z: i32, grid_size: i32) -> i32 {
    return z * grid_size * grid_size + y * grid_size + x;
}

fn other_ray_box_intersection(origin: vec3<f32>, dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> bool {
    var t_min = (box_min - origin) / (dir + 0.00001);
    var t_max = (box_max - origin) / (dir + 0.00001);

    // Correct for negative direction components
    let t1: vec3<f32> = min(t_min, t_max);
    let t2: vec3<f32> = max(t_min, t_max);

    // Find the entry and exit points
    let t_entry: f32 = max(t1.x, max(t1.y, t1.z));
    let t_exit: f32 = min(t2.x, min(t2.y, t2.z));

    // Check for valid intersection
    return t_entry <= t_exit && t_exit >= 0.0;
}


// A WGSL function to calculate the position of a ray after traveling a fixed distance
fn calculateRayPosition(ray_origin: vec3<f32>, ray_direction: vec3<f32>, distance: f32) -> vec3<f32> {
    // Normalize the ray direction to ensure it has a unit length
    let normalized_direction = normalize(ray_direction);

    // Calculate the new position
    let position = ray_origin + normalized_direction * distance;

    return position;
}

// Function to generate the ray direction based on UV, FOV, and aspect ratio
fn generate_ray_direction(
    uv: vec2<f32>,          // UV coordinates, in [0,1]
    fov: f32,               // Field of view (radians)
    aspect_ratio: f32,      // Aspect ratio
    camera_orientation: CameraOrientation // Camera orientation vectors
) -> vec3<f32> {
    // Convert UV to normalized coordinates in the range [-1, 1]
    let uv_centered = uv * 2.0 - vec2<f32>(1.0, 1.0);

    // Account for aspect ratio in x direction
    let x = uv_centered.x * aspect_ratio * tan(fov * 0.5);
    let y = uv_centered.y * tan(fov * 0.5);

    // Generate the ray direction using camera vectors
    let ray_dir = normalize(camera_orientation.cameraFront + x * camera_orientation.cameraRight + y * camera_orientation.cameraUp);

    return ray_dir;
}

fn compute_camera_orientation(yaw: f32, pitch: f32) -> CameraOrientation {
    let front = normalize(vec3<f32>(
        cos(yaw) * cos(pitch),
        sin(pitch),
        sin(yaw) * cos(pitch)
    ));

    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), front));
    let up = cross(front, right);

    return CameraOrientation(front, right, up);
}



struct Camera {
    position: vec3<f32>,   // Camera Position
    yaw: f32,     // FOV check
    pitch: f32,  // Camera Direction
    aspect: f32,         // Aspect ratio
    fov: f32,        // Up Vector
    padding3: f32,         // Padding to fit 48 byte bill
}

@group(1) @binding(0)
var<uniform> camera: Camera; // Binding camera data


@group(0) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;


@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let camera_position = camera.position;//vec3<f32>(0.0, 0.0, -5.0);  // Camera position
    let yaw = radians(camera.yaw);  // Camera yaw (0 degrees)
    let pitch = radians(camera.pitch); // Camera pitch (0 degrees)
    //let fov = radians(camera.fov);  // Field of view in radians
    let fov = camera.fov;
    let aspect_ratio = 16.0 / 9.0; // Aspect ratio (hardcoded to 16:9)
    // Hardcoded screen resolution (change if needed)
    let resolution = vec2<u32>(1920u, 1080u); // Example 16:9 resolution
    let coords = vec2<i32>(global_id.xy);
    let uv = vec2<f32>(coords) / vec2<f32>(resolution); // Normalized UV in [0, 1]
    // Compute the camera orientation based on yaw and pitch
    let camera_orientation = compute_camera_orientation(yaw, pitch);

    // Generate the ray direction using the computed camera orientation
    let ray_dir = generate_ray_direction(uv, fov, aspect_ratio, camera_orientation);
    let ray: Ray = Ray(camera_position, ray_dir);
    


    let sky_position = calculateRayPosition(vec3<f32>(0.0), ray_dir, 1000.0);

    // Visualize the ray direction (e.g., red for hit, blue for no hit)
    let num = 800.0;
    let small_num = 20.0;
    var color: vec4<f32> = vec4<f32>((sky_position.y / num) + 0.5, (sky_position.y / num) + 0.5, 1.0, 1.0); // Default to blue (no hit)
    //color = vec4<f32>((sky_position.y / num), (sky_position.y / num), 1.0 + (sky_position.y / num), 1.0);
    //color = vec4<f32>((sin(sky_position.y / small_num)), (sin(sky_position.y) / small_num), 0.6, 1.0); // cool pattern!


    let hit = trace_tree(ray, vec3<f32>(0.0,0.0,0.0), VOXEL_SCALE);
    //let hit = traverse_64_tree(ray);
    //let box_hit = ray_box_intersection(camera_position, ray_dir, vec3<f32>(-10.0, -10.0, -10.0), vec3<f32>(10.0, 10.0, 10.0));
    //let non = vec4<f32>(0.0);
    if (hit.hit) {
        color = vec4<f32>(hit.color, 1.0);
    } else {
        color = color + vec4<f32>(hit.color, 1.0);
    }


    


    // Store the color in the texture
    textureStore(output_texture, coords, color);

}