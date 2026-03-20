import logging
from typing import AsyncIterable

import rerun as rr
import warp as wp
from newton.examples import _ExampleBrowser, _format_fps
from newton.tests.unittest_utils import find_nan_members

logger = logging.getLogger()
logger.addHandler(rr.LoggingHandler(f"logs/{__name__}"))
logger.setLevel(-1)


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
        # if not viewer.is_running():
            # break
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
