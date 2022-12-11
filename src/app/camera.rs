use std::ops::{Add, AddAssign, Deref, DerefMut, Sub, SubAssign};

use glam::*;
use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};

use super::utils::Dirtiable;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    _Orthographic { w: f32, h: f32 },
    Perspective { fov_y: f32, aspect_ratio: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub pos: Vec3,
    pub rot: Quat,
    pub near: f32,
    pub far: f32,
    pub proj: Projection,
}

impl Camera {
    pub fn as_uniform_data(&self) -> [[f32; 4]; 4] {
        self.view_proj_matrix().to_cols_array_2d()
    }

    /// Creates a camera using an orthographic projection. May be broken right now, I need to
    /// double-check what an orthographic camera should look like
    ///
    /// # Panics
    /// Panics if either of `scale.x` or `scale.y` is 0.
    pub fn _orthographic(pos: Vec3, rot: Quat, near: f32, scale: Vec3) -> Self {
        assert_ne!(scale.x, 0.0);
        assert_ne!(scale.y, 0.0);

        Self {
            pos,
            rot: rot.normalize(),
            near,
            far: near + scale.z,
            proj: Projection::_Orthographic {
                w: scale.x,
                h: scale.y,
            },
        }
    }

    /// `fov_y` is in radians.
    pub fn perspective(
        pos: Vec3,
        rot: Quat,
        near: f32,
        far: f32,
        fov_y: f32,
        aspect_ratio: f32,
    ) -> Self {
        Self {
            pos,
            rot: rot.normalize(),
            near,
            far,
            proj: Projection::Perspective {
                fov_y,
                aspect_ratio,
            },
        }
    }

    /// Converts the camera space to the view space (camera looks at +z, +x is right and
    /// +y is up)
    pub fn view_matrix(&self) -> Mat4 {
        // let t = self.rot.mul_vec3(Vec3::Y);
        // let g = self.rot.mul_vec3(-Vec3::Z);
        // let gxt = g.cross(t);
        // let translate = Mat4::from_translation(-self.pos);
        // let
        // Mat4::from_cols(
        //     // gxt.extend(gxt.dot(self.pos)).into(),
        //     // t.extend(t.dot(self.pos)).into(),
        //     // g.extend(g.dot(self.pos)).into(),
        //     gxt.extend(0.0),
        //     t.extend(0.0),
        //     g.extend(0.0),
        //     [gxt.dot(self.pos), t.dot(self.pos), g.dot(self.pos), 1.0].into()
        // )
        // TODO: this is a fixed multiplication, can figure this out based on g, t, and g x t
        Mat4::from_quat(self.rot.inverse()) * Mat4::from_translation(-self.pos)
    }

    /// Gets this camera's projection matrix, which converts view space coordinates
    /// (camera looks at +z, +x is right, +y is up) to wgpu normalized device coordinates.
    pub fn proj_matrix(&self) -> Mat4 {
        let n = self.near;
        let f = self.far;
        match self.proj {
            Projection::_Orthographic { w, h } => {
                Mat4::orthographic_lh(-w / 2.0, w / 2.0, -h / 2.0, h / 2.0, n, f)
            }
            // Mat4::from_cols(
            //     2.0 / w * Vec4::X,
            //     2.0 / h * Vec4::Y,
            //     2.0 / (f - n) * Vec4::Z,
            //     [0.0, 0.0, (n + f) / (n - f), 1.0].into(),
            // ),
            Projection::Perspective {
                fov_y,
                aspect_ratio,
            } => {
                // i tried to implement it myself and then realized glam already provides methods for various projections
                Mat4::perspective_lh(fov_y, aspect_ratio, n, f)

                // let t = f32::tan(fov_y / 2.0) * self.near.abs();
                // let r = t * aspect_ratio;
                // // this is the transpose of the perspective transform
                // Mat4::from_cols_array_2d(&[
                //     [n / r, 0.0, 0.0, 0.0],
                //     [0.0, n / t, 0.0, 0.0],
                //     [0.0, 0.0, (n + f) / (f - n), 1.0],
                //     [0.0, 0.0, -2.0 * n * f / (n - f), 0.0],
                // ])
            }
        }
    }

    pub fn view_proj_matrix(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// Converts a world/camera space position to normalized device coordinates
    pub fn _transform_point(&self, pos: Vec3) -> Vec3 {
        self.view_proj_matrix().project_point3(pos)
    }

    pub fn look_at(&self, pos: Vec3) -> Self {
        let direction = Vec3::ZERO - Vec3::from(pos);
        let [dx, dy, dz] = direction.to_array();
        let pitch = -dy.atan2(vec2(dx, dz).length());
        let yaw = (-dx).atan2(dz);
        let rot = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        Self { rot, ..*self }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MotionDir {
    Pos,
    Neg,
    None,
}

impl MotionDir {
    pub fn combine(self, rhs: Self) -> Self {
        match self {
            Self::Neg => rhs.dec(),
            Self::None => rhs,
            Self::Pos => rhs.inc(),
        }
    }

    pub fn inv(self) -> Self {
        match self {
            Self::Pos => Self::Neg,
            Self::None => Self::None,
            Self::Neg => Self::Pos,
        }
    }

    pub fn inc(self) -> Self {
        match self {
            Self::Neg => Self::None,
            _ => Self::Pos,
        }
    }
    pub fn dec(self) -> Self {
        match self {
            Self::Pos => Self::None,
            _ => Self::Neg,
        }
    }

    pub fn as_multiplier(self) -> i32 {
        match self {
            Self::Neg => -1,
            Self::None => 0,
            Self::Pos => 1,
        }
    }
}

impl Default for MotionDir {
    fn default() -> Self {
        Self::None
    }
}

impl Add for MotionDir {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.combine(rhs)
    }
}
impl AddAssign for MotionDir {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}
impl Sub for MotionDir {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.combine(rhs.inv())
    }
}
impl SubAssign for MotionDir {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BasicMovement {
    // local coordinates hold the camera looking at +z with +y as up and +x as right
    // they are basically just normalized device coordinates for wgpu
    local_x: MotionDir,
    local_y: MotionDir,
    local_z: MotionDir,
    move_scale: Vec3,
    rot_scale: f32,
}

impl Default for BasicMovement {
    fn default() -> Self {
        Self {
            local_x: Default::default(),
            local_y: Default::default(),
            local_z: Default::default(),
            move_scale: Vec3::ONE,
            rot_scale: 1.0,
        }
    }
}

pub struct BasicCameraController {
    movement: BasicMovement,
    camera: Dirtiable<Camera>,
}

impl BasicCameraController {
    pub fn new(camera: Camera, move_scale: Vec3, rot_scale: f32) -> Self {
        let camera = camera.into();
        Self {
            camera,
            movement: BasicMovement {
                move_scale,
                rot_scale,
                ..Default::default()
            }
            .into(),
        }
    }

    pub fn process_event(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                } => {
                    if let Some((mut x_move, mut y_move, mut z_move)) = match keycode {
                        VirtualKeyCode::Space => {
                            Some((MotionDir::None, MotionDir::Pos, MotionDir::None))
                        }
                        VirtualKeyCode::LShift => {
                            Some((MotionDir::None, MotionDir::Neg, MotionDir::None))
                        }
                        VirtualKeyCode::A => {
                            Some((MotionDir::Neg, MotionDir::None, MotionDir::None))
                        }
                        VirtualKeyCode::D => {
                            Some((MotionDir::Pos, MotionDir::None, MotionDir::None))
                        }
                        VirtualKeyCode::W => {
                            Some((MotionDir::None, MotionDir::None, MotionDir::Pos))
                        }
                        VirtualKeyCode::S => {
                            Some((MotionDir::None, MotionDir::None, MotionDir::Neg))
                        }
                        _ => None,
                    } {
                        if *state == ElementState::Released {
                            x_move = x_move.inv();
                            y_move = y_move.inv();
                            z_move = z_move.inv();
                        }
                        self.movement.local_x += x_move;
                        self.movement.local_y += y_move;
                        self.movement.local_z += z_move;
                    }
                }
                _ => {}
            },
            Event::DeviceEvent { event, .. } => {
                match event {
                    DeviceEvent::MouseMotion { delta } => {
                        let dyaw = (delta.0 as f32).to_radians();
                        let dpitch = (delta.1 as f32).to_radians();
                        let (yaw, pitch, _) = self.camera.rot.to_euler(EulerRot::YXZ);
                        let dyaw = dyaw * self.movement.rot_scale; // * dt;
                        let dpitch = dpitch * self.movement.rot_scale; // * dt;
                                                                       // note: incline is ccw around +x meaning that +theta is down
                                                                       // but you can't just invert it! bc then it oscillates bc you constantly flip signs
                                                                       // you just have to be aware of it e.g. when setting bounds
                                                                       // i could probably find a way to fix it but eh
                        let pitch =
                            (pitch + dpitch).clamp(-60_f32.to_radians(), 85_f32.to_radians());
                        let yaw = yaw + dyaw;
                        self.camera.modify(|camera| {
                            camera.rot = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
                        })
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Updates according to the provided time delta.
    pub fn update_with_dt(&mut self, dt: f32) {
        let movement = &mut self.movement;
        self.camera.modify(|camera| {
            camera.rot = camera.rot.normalize();
            let local_x = (camera.rot.mul_vec3(Vec3::X) * Vec3::X.xyx()).normalize();
            // let local_y = camera.rot.mul_vec3(Vec3::Y);
            let local_z = (camera.rot.mul_vec3(Vec3::Z) * Vec3::X.xyx()).normalize();
            let dpos = dt
                * (local_x * movement.local_x.as_multiplier() as f32
                    + Vec3::Y * movement.local_y.as_multiplier() as f32
                    + local_z * movement.local_z.as_multiplier() as f32);
            camera.pos += dpos;
        });
    }

    /// The semantics of this are a little weird since updates to camera state aren't always
    /// immediately applied.
    pub fn _camera(&self) -> &Dirtiable<Camera> {
        &self.camera
    }
}

impl Deref for BasicCameraController {
    type Target = Dirtiable<Camera>;
    fn deref(&self) -> &Self::Target {
        &self.camera
    }
}

impl DerefMut for BasicCameraController {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.camera
    }
}
