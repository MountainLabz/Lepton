// Vertex shader



@group(1) @binding(0) var voxelTexture: texture_3d<u32>;  // Unsigned 8-bit integer texture
@group(1) @binding(1) var voxelSampler: sampler;


struct IntersectionOutput{
    InPos: f32,
    OutPos: f32,
    Hit: bool,
}

struct CameraOrientation {
    cameraFront: vec3<f32>,
    cameraRight: vec3<f32>,
    cameraUp: vec3<f32>,
};

// Function to get the sign of a number
fn sign(x: f32) -> f32 {
    return select(-1.0, 1.0, x > 0.0);
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




fn get_voxel_value(current_voxel: vec3<i32>, texture_size: vec3<i32>) -> u32 {
    // Normalize voxel coordinates to [0, 1] range
    let normalized_coords: vec3<f32> = vec3<f32>(current_voxel) / vec3<f32>(texture_size);

    // Convert normalized coordinates to integer coordinates
    let int_coords: vec3<i32> = vec3<i32>(normalized_coords * vec3<f32>(texture_size));

    // Sample the 3D texture at the integer coordinates with mip level 0
    let voxel_value: vec4<u32> = textureLoad(voxelTexture, int_coords, 0);

    // Extract the single channel value (assuming it’s stored in the first component)
    return voxel_value.r;  // or voxel_value.x, as all components are the same in a single-channel texture
}

fn get_epsilon(direction: vec3<f32>, voxel_size: vec3<f32>) -> f32 {
    // Determine epsilon based on the direction magnitude and voxel size
    let max_direction = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
    return min(voxel_size.x, min(voxel_size.y, voxel_size.z)) * 0.001 * max_direction;
}


fn DDA_March(
    start_pos: vec3<f32>,  // Starting position of the ray in world space
    end_pos: vec3<f32>,    // Ending position of the ray in world space (not currently used in this function)
    voxel_size: vec3<f32>,       // Size of each voxel in the voxel grid
    bounds: vec3<i32>,     // Dimensions of the voxel grid (width, height, depth)
) -> vec4<f32> {

    var direction: vec3<f32> = normalize(vec3<f32>(end_pos - start_pos));

    //var EPSILON: f32 = get_epsilon(direction, voxel_size);
    // Calculate the direction of the ray
    //

    //


    //var biased_start_pos: vec3<f32> = start_pos + direction * EPSILON;
    //var current_voxel: vec3<i32> = vec3<i32>(floor((biased_start_pos + direction * EPSILON) / voxel_size));
    var current_voxel: vec3<i32> = vec3<i32>(floor(start_pos / voxel_size));

    //var current_voxel: vec3<i32> = vec3<i32>(floor((start_pos - direction * EPSILON) / voxel_size));



    //var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));


    var step: vec3<i32> = vec3<i32>(
    select(-1, 1, direction.x > 0.0),
    select(-1, 1, direction.y > 0.0),
    select(-1, 1, direction.z > 0.0)
    );

    //var current_voxel: vec3<i32> = vec3<i32>(floor((start_pos + direction * EPSILON * vec3<f32>(step)) / voxel_size));

    //var tMax: vec3<f32> = vec3<f32>(
    //    ((floor(start_pos.x / voxel_size.x) + select(1.0, 0.0, direction.x < 0.0)) * voxel_size.x - start_pos.x) / direction.x,
    //   ((floor(start_pos.y / voxel_size.y) + select(1.0, 0.0, direction.y < 0.0)) * voxel_size.y - start_pos.y) / direction.y,
    //    ((floor(start_pos.z / voxel_size.z) + select(1.0, 0.0, direction.z < 0.0)) * voxel_size.z - start_pos.z) / direction.z
    //);

    var tMax: vec3<f32> = vec3<f32>(
        ((select(floor(start_pos.x / voxel_size.x) + 1.0, floor(start_pos.x / voxel_size.x), direction.x < 0.0) * voxel_size.x) - start_pos.x) / direction.x,
        ((select(floor(start_pos.y / voxel_size.y) + 1.0, floor(start_pos.y / voxel_size.y), direction.y < 0.0) * voxel_size.y) - start_pos.y) / direction.y,
        ((select(floor(start_pos.z / voxel_size.z) + 1.0, floor(start_pos.z / voxel_size.z), direction.z < 0.0) * voxel_size.z) - start_pos.z) / direction.z
    );


    var tDelta: vec3<f32> = vec3<f32>(
        abs(voxel_size.x / direction.x),
        abs(voxel_size.y / direction.y),
        abs(voxel_size.z / direction.z)
    );


    while(within_bounds(current_voxel, bounds)){
        var voxel = get_voxel_value(vec3<i32>(current_voxel), bounds);
        var world_cord = vec3<f32>(vec3<f32>(current_voxel) * voxel_size);

        if (voxel == 1){
            var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));
            return vec4<f32>(dist / 30.0, dist / 30.0,dist / 30.0, 0.0);
        }

        if (voxel == 2){
            var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));
            return vec4<f32>(dist / 30.0, dist / 30.0 + 0.3,dist / 30.0, 0.0);
        }

        if (voxel == 3){
            var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));
            return vec4<f32>(dist / 30.0, (dist / 30.0) + 0.2,dist / 30.0, 0.0);
        }
        if (voxel == 4){
            var dist: f32 = f32(sqrt(((camera.position.x - world_cord.x) * (camera.position.x - world_cord.x)) + ((camera.position.y - world_cord.y) * (camera.position.y - world_cord.y)) + ((camera.position.z - world_cord.z) * (camera.position.z - world_cord.z))));
            return vec4<f32>((dist / 30.0) + 0.31, (dist / 30.0) + 0.23,dist / 30.0, 0.0);
        }


        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            current_voxel.x += step.x;
            tMax.x += tDelta.x;
        } else if (tMax.y < tMax.z) {
            current_voxel.y += step.y;
            tMax.y += tDelta.y;
        } else {
            current_voxel.z += step.z;
            tMax.z += tDelta.z;
        }


        direction = normalize(direction);
        //if (tMax.x < tMax.y && tMax.x < tMax.z){
        //    current_voxel.x += step.x;
        //    tMax.x += tDelta.x;
        //} else if (tMax.y < tMax.z){
        //    current_voxel.y += step.y;
        //    tMax.y += tDelta.y;
        //} else{
        //    current_voxel.z += step.z;
        //    tMax.z += tDelta.z;
        //}





    }


    return vec4<f32>(0.0,0.0,0.0,0.0);
}









fn compute_camera_orientation(pitch: f32, yaw: f32) -> CameraOrientation {
    // Compute the cameraFront vector from yaw and pitch
    var x = cos(yaw) * cos(pitch);
    var y = sin(pitch);
    var z = sin(yaw) * cos(pitch);
    let cameraFront = normalize(vec3<f32>(x, y, z));

    // Compute the cameraRight vector as the cross product of a world-up vector (0, 1, 0) and cameraFront
    let worldUp = vec3<f32>(0.0, 1.0, 0.0);
    let cameraRight = normalize(cross(worldUp, cameraFront));

    // Compute the cameraUp vector as the cross product of cameraFront and cameraRight
    let cameraUp = normalize(cross(cameraRight, cameraFront));

    return CameraOrientation(cameraFront, cameraRight, cameraUp);
}


fn generate_ray_direction(
    uv: vec2<f32>,           // UV coordinates (0,0) is bottom-left and (1,1) is top-right
    pitch: f32,              // Camera pitch (up-down rotation in radians)
    yaw: f32,                // Camera yaw (left-right rotation in radians)
    fov: f32,                // Field of view in radians
    aspectRatio: f32         // Aspect ratio (width / height of the viewport)
) -> vec3<f32> {           // Returns the ray direction
    // Compute camera orientation vectors from pitch and yaw
    let CO = compute_camera_orientation(pitch, yaw);
    let cameraFront = CO.cameraFront;
    let cameraRight = CO.cameraRight;
    let cameraUp = CO.cameraUp;

    // Convert UV coordinates to normalized device coordinates (-1, 1)
    let x = (uv.x * 2.0 - 1.0) * aspectRatio;  // Map UV x to range [-aspectRatio, aspectRatio]
    let y = (uv.y * 2.0 - 1.0);                // Map UV y to range [-1, 1]

    // Calculate the ray direction in camera space
    // The tan(fov * 0.5) scales the direction based on FOV
    let rayDir = normalize(
        cameraFront + x * tan(fov * 0.5) * cameraRight + y * tan(fov * 0.5) * cameraUp
    );

    return rayDir;
}

// Function to test ray-sphere intersection
fn ray_sphere_intersection(
    rayOrigin: vec3<f32>,  // Origin of the ray
    rayDir: vec3<f32>,     // Direction of the ray (normalized)
    radius: f32            // Radius of the sphere
) -> bool {              // Returns true if there is an intersection, false otherwise
    // Compute the vector from the ray origin to the sphere center (which is at (0,0,0))
    let oc = rayOrigin;

    // Compute coefficients of the quadratic equation: t^2 * dot(rayDir, rayDir) + 2 * t * dot(oc, rayDir) + dot(oc, oc) - radius^2 = 0
    let a = dot(rayDir, rayDir);  // This is usually 1.0 if rayDir is normalized
    let b = 2.0 * dot(oc, rayDir);
    let c = dot(oc, oc) - radius * radius;

    // Compute the discriminant
    let discriminant = b * b - 4.0 * a * c;

    // Check if there is an intersection
    return discriminant >= 0.0;  // True if there is an intersection, false otherwise
}



// Ray-box intersection function using the slab method
fn ray_box_intersection(ray_origin: vec3<f32>, ray_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> IntersectionOutput {
    let inv_dir = 1.0 / ray_dir;

    // Calculate intersection t-values for the minimum and maximum faces of the box
    let t_min = (box_min - ray_origin) * inv_dir;
    let t_max = (box_max - ray_origin) * inv_dir;

    // Calculate the entry and exit points by taking the minimum and maximum t-values
    let t1 = min(t_min, t_max);
    let t2 = max(t_min, t_max);

    // Determine the entry and exit points along the ray
    let t_near = max(max(t1.x, t1.y), t1.z);
    let t_far = min(min(t2.x, t2.y), t2.z);

    // Check if there is an intersection
    let hit: bool = (t_near <= t_far) && (t_far >= 0.0);

    // We could use `hit` later or store it in a global variable if needed

    return IntersectionOutput(t_near, t_far, hit);  // t_near is entry, t_far is exit
}


struct Camera {
    position: vec3<f32>,   // Camera Position
    fov: f32,         // FOV check
    direction: vec3<f32>,  // Camera Direction
    aspect: f32,         // Aspect ratio
    up: vec3<f32>,         // Up Vector
    padding3: f32,         // Padding to fit 48 byte bill
};

@group(0) @binding(0)
var<uniform> camera: Camera; // Binding camera data





struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = vec4<f32>(model.position, 1.0);
    out.uv = model.uv;
    return out;
}



// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
// Example camera parameters

    var pitch: f32 = radians(camera.direction.y); // Convert pitch to radians
    var yaw: f32 = radians(camera.direction.x); // Convert yaw to radians (looking along -Z axis)
    var fov: f32 = radians(camera.fov);
    var aspect: f32 = 16.0 / 9.0;
    // Compute ray direction based on camera parameters and UV
    var rayDir = generate_ray_direction(in.uv, pitch, yaw, camera.fov, aspect);

    // Use rayDir for ray tracing or rendering

    var box_min: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var box_max: vec3<f32> = vec3<f32>(10.0, 10.0, 10.0);




    var results: IntersectionOutput = ray_box_intersection(camera.position, rayDir, box_min, box_max);
    var hit = results.Hit;
    var InPos = camera.position + (vec3<f32>(results.InPos) * rayDir);
    var OutPos = camera.position + (vec3<f32>(results.OutPos) * rayDir);

    //current problems are likely to do with voxel position calculation and actually marching through them properly.


    if(hit == true){
        //return vec4<f32>(InPos, 0.0);
        var color = DDA_March(InPos, OutPos, vec3<f32>(0.1), vec3<i32>(100,100,100));
        if (all(color == vec4<f32>(0.0,0.0,0.0,0.0))){
            var position: vec3<f32> = vec3<f32>(camera.position + (normalize(rayDir) * vec3<f32>(50.0)));
            return vec4<f32>((position.y / 50.0), (position.y / 50.0), 1.0, 0.0);
        }
        return color;
    } else{
        var position: vec3<f32> = vec3<f32>(camera.position + (normalize(rayDir) * vec3<f32>(50.0)));
        return vec4<f32>((position.y / 50.0), (position.y / 50.0), 1.0, 0.0);
    }



}
