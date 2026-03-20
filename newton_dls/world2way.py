import sys

import newton
import newton.examples
import newton.utils
import numpy as np
import torch
import warp as wp
from newton.examples.robot.example_robot_anymal_c_walk import (
    compute_obs,
    lab_to_mujoco,
    mujoco_to_lab,
)
from newton.solvers import SolverImplicitMPM


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    cid = collider_ids[i]

    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return

        f_world = collider_impulses[i] / dt
        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)

        wp.atomic_add(
            body_f,
            body_index,
            wp.spatial_vector(f_world, wp.cross(r, f_world)),
        )


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    body_id = wp.tid()

    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)

    r = wp.transform_get_rotation(body_q[body_id])
    delta_w = dt * wp.quat_rotate(
        r,
        body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)),
    )

    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


class Example:
    def __init__(self, viewer, args):
        voxel_size = args.voxel_size
        particles_per_cell = args.particles_per_cell
        tolerance = args.tolerance
        grid_type = args.grid_type

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.device = wp.get_device()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        builder.request_state_attributes("mujoco:qfrc_actuator")

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(asset_path / "urdf" / "anymal.urdf")
        builder.add_urdf(
            stage_path,
            xform=wp.transform(
                wp.vec3(0.0, -1.3, 1.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            ),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        # Only shanks collide with particles.
        # If the interaction is too weak, also enable feet or more links.
        for body in range(builder.body_count):
            if "SHANK" not in builder.body_label[body]:
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] = (
                        builder.shape_flags[shape]
                        & ~newton.ShapeFlags.COLLIDE_PARTICLES
                    )

        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, 0.0, -0.500), wp.quat_identity()),
            hx=5.0,
            hy=5.0,
            hz=0.5,
            label="ground_box",
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((1.5, 0.0, 0.2), wp.quat_identity()),
            hx=1.0,
            hy=3.0,
            hz=0.2,
            label="box1",
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((-1.5, 0.0, 0.2), wp.quat_identity()),
            hx=1.0,
            hy=3.0,
            hz=0.2,
            label="box2",
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, -1.8, 0.2), wp.quat_identity()),
            hx=0.5,
            hy=1.2,
            hz=0.2,
            label="box3",
        )

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 50
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        initial_q = {
            "RH_HAA": 0.0,
            "RH_HFE": -0.4,
            "RH_KFE": 0.8,
            "LH_HAA": 0.0,
            "LH_HFE": -0.4,
            "LH_KFE": 0.8,
            "RF_HAA": 0.0,
            "RF_HFE": 0.4,
            "RF_KFE": -0.8,
            "LF_HAA": 0.0,
            "LF_HFE": 0.4,
            "LF_KFE": -0.8,
        }
        for name, value in initial_q.items():
            idx = next(
                i
                for i, lbl in enumerate(builder.joint_label)
                if lbl.endswith(f"/{name}")
            )
            builder.joint_q[idx + 6] = value

        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        SolverImplicitMPM.register_custom_attributes(builder)

        density = 2500.0
        particle_lo = np.array([-0.5, -0.6, 0.0])
        particle_hi = np.array([0.5, 2.5, 0.4])
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )
        _spawn_particles(builder, particle_res, particle_lo, particle_hi, density)

        self.model = builder.finalize()

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = tolerance
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = grid_type
        mpm_options.grid_padding = 50 if grid_type == "fixed" else 0
        mpm_options.max_active_cell_count = 1 << 15 if grid_type == "fixed" else -1
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 1.0
        mpm_options.collider_velocity_mode = "finite_difference"

        self.model.mpm.hardening.fill_(0.6)
        self.model.mpm.damping.fill_(0.6)
        self.model.mpm.friction.fill_(0.4)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            ls_iterations=50,
            njmax=50,
        )
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )

        # Two-way coupling: use the actual model as collider source.
        self.mpm_solver.setup_collider(model=self.model)

        max_nodes = 1 << 20
        self.collider_impulses = wp.zeros(
            max_nodes, dtype=wp.vec3, device=self.model.device
        )
        self.collider_impulse_pos = wp.zeros(
            max_nodes, dtype=wp.vec3, device=self.model.device
        )
        self.collider_impulse_ids = wp.full(
            max_nodes, value=-1, dtype=int, device=self.model.device
        )
        self.collider_body_id = self.mpm_solver.collider_body_index
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)
        self.body_q_mpm = wp.empty_like(self.state_0.body_q)
        self.body_qd_mpm = wp.empty_like(self.state_0.body_qd)
        self.collect_collider_impulses()

        self.control = self.model.control()

        q0 = wp.to_torch(self.state_0.joint_q)
        self.torch_device = q0.device
        self.joint_pos_initial = q0[7:].unsqueeze(0).detach().clone()
        self.act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(
            1, 12, device=self.torch_device, dtype=torch.float32
        )

        policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")
        self.policy = torch.jit.load(policy_path, map_location=self.torch_device)

        self.lab_to_mujoco_indices = torch.tensor(
            [lab_to_mujoco[i] for i in range(len(lab_to_mujoco))],
            device=self.torch_device,
        )
        self.mujoco_to_lab_indices = torch.tensor(
            [mujoco_to_lab[i] for i in range(len(mujoco_to_lab))],
            device=self.torch_device,
        )
        self.gravity_vec = torch.tensor(
            [0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32
        ).unsqueeze(0)
        self.command = torch.zeros(
            (1, 3), device=self.torch_device, dtype=torch.float32
        )

        self._auto_forward = True

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.capture()

    def capture(self):
        # Keep capture disabled until the two-way state swapping is validated.
        self.graph = None
        self.sand_graph = None

    def collect_collider_impulses(self):
        collider_impulses, collider_impulse_pos, collider_impulse_ids = (
            self.mpm_solver._collect_collider_impulses(self.state_0)
        )

        self.collider_impulse_ids.fill_(-1)
        n = min(collider_impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n].assign(collider_impulses[:n])
        self.collider_impulse_pos[:n].assign(collider_impulse_pos[:n])
        self.collider_impulse_ids[:n].assign(collider_impulse_ids[:n])

    def apply_control(self):
        obs = compute_obs(
            self.act,
            self.state_0,
            self.joint_pos_initial,
            self.torch_device,
            self.lab_to_mujoco_indices,
            self.gravity_vec,
            self.command,
        )
        with torch.no_grad():
            self.act = self.policy(obs)
            self.rearranged_act = torch.gather(
                self.act, 1, self.mujoco_to_lab_indices.unsqueeze(0)
            )
            a = self.joint_pos_initial + 0.5 * self.rearranged_act
            a_with_zeros = torch.cat(
                [
                    torch.zeros(6, device=self.torch_device, dtype=torch.float32),
                    a.squeeze(0),
                ]
            )
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            wp.copy(self.control.joint_target_pos, a_wp)

    def simulate_robot(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    self.frame_dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.state_0.body_f,
                ],
            )

            # Save only the force coming from the previous MPM step.
            self.body_sand_forces.assign(self.state_0.body_f)

            self.viewer.apply_forces(self.state_0)
            self.solver.step(
                self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        body_q_real = self.state_0.body_q
        body_qd_real = self.state_0.body_qd

        wp.launch(
            subtract_body_force,
            dim=body_q_real.shape[0],
            inputs=[
                self.frame_dt,
                body_q_real,
                body_qd_real,
                self.body_sand_forces,
                self.model.body_inv_inertia,
                self.model.body_inv_mass,
                self.body_q_mpm,
                self.body_qd_mpm,
            ],
        )

        self.state_0.body_q = self.body_q_mpm
        self.state_0.body_qd = self.body_qd_mpm

        self.mpm_solver.step(
            self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt
        )

        self.state_0.body_q = body_q_real
        self.state_0.body_qd = body_qd_real

        self.collect_collider_impulses()

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            fwd = (
                1.0
                if self.viewer.is_key_down("i")
                else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            )
            lat = (
                0.5
                if self.viewer.is_key_down("j")
                else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            )
            rot = (
                1.0
                if self.viewer.is_key_down("u")
                else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            )

            if fwd or lat or rot:
                self._auto_forward = False

            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)

        if self._auto_forward:
            self.command[0, 0] = 1

        self.apply_control()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate_robot()

        if self.sand_graph:
            wp.capture_launch(self.sand_graph)
        else:
            self.simulate_sand()

        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.1,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot went in the right direction",
            lambda q, qd: q[1] > 0.9,
        )

        forward_vel_min = wp.spatial_vector(-0.2, 0.9, -0.2, -0.8, -1.5, -0.5)
        forward_vel_max = wp.spatial_vector(0.2, 1.1, 0.2, 0.8, 1.5, 0.5)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot is moving forward and not falling",
            lambda q, qd: newton.math.vec_inside_limits(
                qd, forward_vel_min, forward_vel_max
            ),
            indices=[0],
        )
        voxel_size = self.mpm_solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)
        parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
        parser.add_argument(
            "--grid-type", "-gt", choices=["sparse", "dense", "fixed"], default="sparse"
        )
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
        return parser


def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, density):
    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = np.prod(cell_size)
    radius = np.max(cell_size) * 0.5
    mass = np.prod(cell_volume) * density

    builder.add_particle_grid(
        pos=wp.vec3(bounds_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=res[0] + 1,
        dim_y=res[1] + 1,
        dim_z=res[2] + 1,
        cell_x=cell_size[0],
        cell_y=cell_size[1],
        cell_z=cell_size[2],
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
    )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    if wp.get_device().is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    example = Example(viewer, args)
    newton.examples.run(example, args)
