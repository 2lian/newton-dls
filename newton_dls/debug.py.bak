from contextlib import suppress
import importlib
import os
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import AsyncIterable

import asyncio_for_robotics as afor
import newton
import newton.examples
import numpy as np
import rerun as rr
import warp as wp
from asyncio_for_robotics.core.sub import asyncio
from newton.examples import _ExampleBrowser, _format_fps
from newton.examples.robot import example_robot_anymal_c_walk as newt_example
from newton.tests.unittest_utils import find_nan_members


async def arun(example, async_looper: AsyncIterable, args):
    """Copied and modified from newton

    /home/elian/newton-dls/deps/newton/newton/examples/__init__.py

    Args:
        example (): The thing to run
        args ():
        async_looper: at each iteration, the sim loop executes

    Raises:
        NotImplementedError:
        ValueError:
    """
    viewer = example.viewer
    example_class = type(example)

    perform_test = args is not None and args.test
    test_post_step = perform_test and hasattr(example, "test_post_step")
    test_final = perform_test and hasattr(example, "test_final")

    browser = _ExampleBrowser(viewer) if not perform_test else None

    if hasattr(example, "gui") and hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(lambda ui, ex=example: ex.gui(ui), position="side")

    async for _ in async_looper:
        if not viewer.is_running():
            break
        if browser is not None and browser.switch_target is not None:
            example, example_class = browser.switch(example_class)
            continue

        if browser is not None and browser._reset_requested:
            example = browser.reset(example_class)
            continue

        if example is None:
            viewer.begin_frame(0.0)
            viewer.end_frame()
            continue

        if not viewer.is_paused():
            with wp.ScopedTimer("step", active=False):
                example.step()
        if test_post_step:
            example.test_post_step()

        with wp.ScopedTimer("render", active=False):
            example.render()

    if perform_test:
        if test_final:
            example.test_final()
        elif not (test_post_step or test_final):
            raise NotImplementedError(
                "Example does not have a test_final or test_post_step method"
            )

    viewer.close()

    if hasattr(viewer, "benchmark_result"):
        result = viewer.benchmark_result()
        if result is not None:
            print(
                f"Benchmark: {_format_fps(result['fps'])} FPS ({result['frames']} frames in {result['elapsed']:.2f}s)"
            )

    if perform_test:
        # generic tests for finiteness of Newton objects
        if hasattr(example, "state_0"):
            nan_members = find_nan_members(example.state_0)
            if nan_members:
                raise ValueError(f"NaN members found in state_0: {nan_members}")
        if hasattr(example, "state_1"):
            nan_members = find_nan_members(example.state_1)
            if nan_members:
                raise ValueError(f"NaN members found in state_1: {nan_members}")
        if hasattr(example, "model"):
            nan_members = find_nan_members(example.model)
            if nan_members:
                raise ValueError(f"NaN members found in model: {nan_members}")
        if hasattr(example, "control"):
            nan_members = find_nan_members(example.control)
            if nan_members:
                raise ValueError(f"NaN members found in control: {nan_members}")
        if hasattr(example, "contacts"):
            nan_members = find_nan_members(example.contacts)
            if nan_members:
                raise ValueError(f"NaN members found in contacts: {nan_members}")


class InjectedExample(newt_example.Example):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.step_sub = afor.BaseSub()

    def step(self):
        super().step()
        # good for non-multithreaded
        self.step_sub._input_data_asyncio(self)


async def log_motors(sim: InjectedExample):
    async for _ in sim.step_sub.listen():
        rr.log("/robot/joint/q", rr.Scalars(sim.state_0.joint_q.list()))
        rr.log("/robot/joint/qd", rr.Scalars(sim.state_0.joint_qd.list()))


def setup_rerun():
    rr.init("newton-dls")
    rr.save("data.rrd")
    rr.connect_grpc()
    # rr.set_time("sim_time", duration=0)
    # rr.set_time("sim_step", sequence=0)
    try:
        pass
    except RuntimeError:
        print(
            "\n\n /!\\ \n Start rerun FIRST using: `pixi run rerun ./rerun_template.rbl` \n Then restart this sim to see the data live. \n The archive `data.rrd` cannot be opened for some reasons. \n /!\\ \n\n"
        )


async def main(viewer, args):
    example = InjectedExample(viewer, args)
    setup_rerun()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(log_motors(example))
        await arun(example, afor.Rate(1 / example.frame_dt).listen(), args)
        for t in tg._tasks:
            t.cancel()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        parser = newt_example.Example.create_parser()
        viewer, args = newton.examples.init(parser)

        asyncio.run(main(viewer, args))
