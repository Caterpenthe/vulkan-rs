use glam::*;

pub enum CameraType {
    LookAt,
    FirstPerson,
}

pub struct ViewPerspective {
    pub view: Mat4,
    pub perspective: Mat4,
}

pub struct Keys {
    pub(crate) left: bool,
    pub(crate) right: bool,
    pub(crate) up: bool,
    pub(crate) down: bool,
}

pub struct Camera {
    fov: f32,
    znear: f32,
    zfar: f32,

    rotation: Vec3,
    position: Vec3,
    view_pos: Vec4,

    flip_y: bool,

    pub camera_type: CameraType,
    pub matrices: ViewPerspective,

    rotation_speed: f32,
    movement_speed: f32,

    updated: bool,

    pub(crate) keys: Keys,
}

impl Camera {
    pub fn new() -> Self {
        Camera {
            fov: 0.0,
            znear: 0.0,
            zfar: 0.0,
            rotation: Default::default(),
            position: Default::default(),
            view_pos: Default::default(),
            flip_y: false,
            camera_type: CameraType::LookAt,
            matrices: ViewPerspective {
                view: Default::default(),
                perspective: Default::default(),
            },
            rotation_speed: 0.0,
            movement_speed: 0.0,
            updated: false,
            keys: Keys {
                left: false,
                right: false,
                up: false,
                down: false,
            },
        }
    }
    pub fn moving(&mut self) -> bool {
        self.keys.left || self.keys.right || self.keys.up || self.keys.down
    }
    pub fn set_perspective(&mut self, fov: f32, aspect: f32, znear: f32, zfar: f32) {
        self.fov = fov;
        self.znear = znear;
        self.zfar = zfar;
        self.matrices.perspective = Mat4::perspective_rh(f32::to_radians(fov), aspect, znear, zfar);
        if self.flip_y {
            self.matrices.perspective.y_axis.y *= -1.0;
        }
    }
    pub fn update_aspect_ratio(&mut self, aspect: f32) {
        self.matrices.perspective =
            Mat4::perspective_lh(f32::to_radians(self.fov), aspect, self.znear, self.zfar);
        if self.flip_y {
            self.matrices.perspective.y_axis.y *= -1.0;
        }
    }

    pub fn set_rotation(&mut self, rotation: Vec3) {
        self.rotation = rotation;
        self.update_view_matrix();
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
        self.update_view_matrix();
    }

    pub fn rotate(&mut self, delta: Vec3) {
        self.rotation += delta;
        self.update_view_matrix();
    }

    pub fn set_translation(&mut self, translation: Vec3) {
        self.position = translation;
        self.update_view_matrix();
    }

    pub fn translate(&mut self, delta: Vec3) {
        self.position += delta;
        self.update_view_matrix();
    }

    pub fn set_rotation_speed(&mut self, speed: f32) {
        self.rotation_speed = speed;
    }

    pub fn set_movement_speed(&mut self, speed: f32) {
        self.movement_speed = speed;
    }

    pub fn update(&mut self, delta_time: f32) {
        self.updated = false;
        if let CameraType::FirstPerson = self.camera_type {
            if self.moving() {
                let mut velocity = Vec3::ZERO;
                if self.keys.left {
                    velocity.x -= self.movement_speed * delta_time;
                }
                if self.keys.right {
                    velocity.x += self.movement_speed * delta_time;
                }
                if self.keys.up {
                    velocity.z -= self.movement_speed * delta_time;
                }
                if self.keys.down {
                    velocity.z += self.movement_speed * delta_time;
                }

                if self.flip_y {
                    velocity.z *= -1.0;
                }

                let rot_mat = Mat4::from_rotation_y(self.rotation.y);
                self.position += rot_mat.transform_vector3(velocity);
                self.updated = true;
            }
        }
        self.update_view_matrix();
    }

    /// Update camera passing separate axis data (gamepad)
    /// Returns true if the view or position has been changed
    pub fn update_pad(axis_left: Vec2, axis_right: Vec2, delta_time: f32) {
        todo!()
    }

    fn update_view_matrix(&mut self) {
        let mut rot_matrix = Mat4::IDENTITY;

        let mut trans_matrix = Mat4::ZERO;

        rot_matrix = rot_matrix
            * Mat4::from_rotation_x(self.rotation.x * if self.flip_y { -1.0 } else { 1.0 });
        rot_matrix = rot_matrix * Mat4::from_rotation_y(self.rotation.y);
        rot_matrix = rot_matrix * Mat4::from_rotation_z(self.rotation.z);

        let mut translation = self.position;
        if self.flip_y {
            translation.y *= -1.0;
        }
        trans_matrix = Mat4::from_translation(translation);
        match self.camera_type {
            CameraType::FirstPerson => {
                self.matrices.view = rot_matrix * trans_matrix;
            }
            CameraType::LookAt => {
                self.matrices.view = trans_matrix * rot_matrix;
            }
        }

        self.view_pos = Vec4::new(self.position.x, self.position.y, self.position.z, 1.0)
            * Vec4::new(-1.0, 1.0, -1.0, 1.0);
    }
}
