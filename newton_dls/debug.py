import asyncio
import logging
from contextlib import suppress

import asyncio_for_robotics as afor
import newton
import newton.examples
import numpy as np
import quaternion as qt
import rerun as rr

# from newton.examples.robot import example_robot_anymal_c_walk as newt_example
from numpy.typing import NDArray

from . import world as sand_example
from .mesh_stuff import newton_mesh_to_rerun
from .utils import arun

logger = logging.getLogger()
logger.addHandler(rr.LoggingHandler(f"logs/{__name__}"))
logger.setLevel(-1)


class InjectedExample(sand_example.Example):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.step_sub = afor.BaseSub()

    def simulate(self):
        return super().simulate()

    def step(self):
        super().step()
        rr.set_time("sim_step", sequence=self.sim_step)
        rr.set_time("sim_time", duration=self.sim_time)
        log_motors(self)
        log_particules(self)


def log_motors(sim: InjectedExample):

    q: NDArray = sim.state_0.joint_q.numpy()
    qd: NDArray = sim.state_0.joint_qd.numpy()
    # you need to add this to the sim builder:
    # builder.request_state_attributes("mujoco:qfrc_actuator")
    qf: NDArray = sim.state_0.mujoco.qfrc_actuator.list()
    # qf: NDArray = sim.control.joint_f.numpy()
    for i in range(sim.model.joint_count):
        sim.model.joint_label
        rr.log(
            f"/robot/joint/q/{sim.model.joint_label[i]}",
            rr.Scalars(q[i]),
        )
        rr.log(
            f"/robot/joint/qd/{sim.model.joint_label[i]}",
            rr.Scalars(qd[i]),
        )
        rr.log(
            f"/robot/joint/qf/{sim.model.joint_label[i]}",
            rr.Scalars(qf[i]),
        )

    tfs: NDArray = sim.state_0.body_q.numpy()
    frames = sim.model.body_label
    for i, name in enumerate(frames):
        quat = qt.from_float_array(tfs[i, 3:])
        frame_name = f"/tf/{name}"
        rr.log(
            frame_name,
            rr.Transform3D(
                translation=tfs[i, :3],
                quaternion=tfs[i, 3:],
            ),
            rr.TransformAxes3D(axis_length=0.1),
        )


def log_particules(sim: InjectedExample):
    particules = sim.state_0.particle_q.numpy()
    particules_vel = sim.state_0.particle_qd.numpy()
    # particules = sim.state_0.particle_f.numpy()
    rr.log(
        "/particules/tf",
        rr.Points3D(particules, radii=rr.Radius(0.005), colors=[200, 170, 120, 255]),
    )
    rr.log(
        "/particules/vel",
        rr.Arrows3D(
            vectors=particules_vel / 10,
            origins=particules,
            radii=rr.Radius.ui_points(0.5),
        ),
    )


def log_mesh(sim: InjectedExample):
    assert sim.model.shape_source is not None
    assert sim.model.shape_label is not None
    body_to_shapes: dict[int, list[int]] = sim.model.body_shapes
    for index, ntn_mesh_obj, name in zip(
        range(sim.model.shape_count), sim.model.shape_source, sim.model.shape_label
    ):
        logger.info(
            f"Mesh: {index=} {name=} {sim.model.shape_type.numpy()[index]=} {sim.model.shape_scale.numpy()[index]=} {ntn_mesh_obj=}"
        )
        is_box = sim.model.shape_type.numpy()[index] == 7

        if ntn_mesh_obj is None:
            if not is_box:
                continue
        for body_ind, shapes in body_to_shapes.items():
            if body_ind == -1:
                continue
            if index in shapes:
                body_name = sim.model.body_label[body_ind]
                name = f"{body_name}/{name}"
        ntn_mesh: newton.Mesh = ntn_mesh_obj  # type: ignore
        if not is_box:
            rr_m = newton_mesh_to_rerun(ntn_mesh)
        else:
            rr_m = rr.Boxes3D(sizes=sim.model.shape_scale.numpy()[index] * 2)
        rr.log(
            f"/tf/{name}",
            rr_m,
            rr.Transform3D(
                translation=sim.model.shape_transform.numpy()[index, :3],
                quaternion=sim.model.shape_transform.numpy()[index, 3:],
            ),
            static=True,
        )


def setup_rerun():
    rr.init("newton-dls")
    rr.save("data.rrd")
    rr.connect_grpc()
    logger.debug("Rerun started")
    rr.set_time("sim_time", duration=0)
    rr.set_time("sim_step", sequence=0)
    try:
        pass
    except RuntimeError:
        print(
            "\n\n /!\\ \n Start rerun FIRST using: `pixi run rerun ./rerun_template.rbl` \n Then restart this sim to see the data live. \n The archive `data.rrd` cannot be opened for some reasons. \n /!\\ \n\n"
        )


async def main(viewer, args):
    setup_rerun()
    logger.info("Instantiating Simulation")
    example = InjectedExample(viewer, args)
    logger.info("Simulation instantiated")
    log_mesh(example)

    async with asyncio.TaskGroup() as tg:
        # tg.create_task(log_motors(example))
        await arun(example, afor.Rate(1 / example.frame_dt).listen(), args)
        logger.debug("Loop started")
        for t in tg._tasks:
            t.cancel()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        parser = sand_example.Example.create_parser()
        viewer, args = newton.examples.init(parser)

        asyncio.run(main(viewer, args))
