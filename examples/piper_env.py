"""Minimal robot-environment adapter for the AgileX (PiperX) bimanual setup.

DSRL only needs three things from the environment:

    reset()             -> move the arms to a known start pose
    get_observation()   -> return the current cameras + proprio
    step(action)        -> command a single 14-D joint+gripper target

Wire-up options (pick one):

1. **Factory (recommended)** — point at your existing piperx_lerobot_setup code::

       export PIPER_ENV_FACTORY=piperx_lerobot_setup.env:make_piper_env

   ``make_piper_env(reset_pose=None)`` must return an object with ``reset()``,
   ``get_observation()``, and ``step(action)`` (see Backend protocol below).

2. **Inline hooks** — edit the four ``_robot_*`` methods at the bottom of this file.

Action / state layout (must match the LeRobot dataset + openpi config):

    index  0..5   left arm joints (rad)
    index  6      left gripper (meters, ~0 closed .. ~0.07 open)
    index  7..12  right arm joints (rad)
    index  13     right gripper (meters)
"""

import importlib
import os
import time
from typing import Any, Callable, Optional

import numpy as np


CAM_FRONT = "cam_front"
CAM_LEFT_WRIST = "cam_left_wrist"
CAM_RIGHT_WRIST = "cam_right_wrist"
CAMERA_NAMES = (CAM_FRONT, CAM_LEFT_WRIST, CAM_RIGHT_WRIST)

STATE_DIM = 14


def _load_callable(spec: str) -> Callable[..., Any]:
    """Import ``module.path:attr`` (function or class)."""
    module_name, attr_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _try_load_backend(reset_pose, robot) -> Optional[Any]:
    """Load a full env backend from ``PIPER_ENV_FACTORY`` if set."""
    spec = os.environ.get("PIPER_ENV_FACTORY")
    if not spec:
        return None
    factory = _load_callable(spec)
    if robot is not None:
        return robot
    try:
        return factory(reset_pose=reset_pose)
    except TypeError:
        return factory()


class PiperEnv:
    """Thin wrapper around piperx_lerobot_setup for DSRL online rollouts."""

    def __init__(self, control_hz=30, reset_pose=None, robot=None):
        self.control_hz = control_hz
        self.reset_pose = reset_pose

        self._backend = _try_load_backend(reset_pose, robot)
        if self._backend is not None:
            self._use_backend = True
            return

        self._use_backend = False
        self._robot = robot
        if self._robot is None:
            self._robot = self._robot_connect()

    # ------------------------------------------------------------------ #
    # DSRL-facing API
    # ------------------------------------------------------------------ #
    def reset(self):
        if self._use_backend:
            return self._backend.reset()

        self._robot_go_to_reset()
        time.sleep(0.5)
        return self.get_observation()

    def get_observation(self):
        if self._use_backend:
            obs = self._backend.get_observation()
            _validate_observation(obs)
            return obs

        images = self._robot_read_cameras()
        for name in CAMERA_NAMES:
            if name not in images:
                raise KeyError(
                    f"Camera '{name}' missing from robot observation; got {list(images)}"
                )
        state = np.asarray(self._robot_read_state(), dtype=np.float32)
        assert state.shape == (STATE_DIM,), f"state must be (14,), got {state.shape}"
        return {"images": images, "state": state}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(STATE_DIM)
        if self._use_backend:
            self._backend.step(action)
            return
        self._robot_send_action(action)

    # ------------------------------------------------------------------ #
    # Hooks (only used when PIPER_ENV_FACTORY is not set)
    # ------------------------------------------------------------------ #
    def _robot_connect(self):
        raise NotImplementedError(
            "Set PIPER_ENV_FACTORY=your.module:make_piper_env (recommended), or "
            "implement PiperEnv._robot_connect in examples/piper_env.py.\n"
            "Example:\n"
            "  export PIPER_ENV_FACTORY=piperx_lerobot_setup.env:make_piper_env"
        )

    def _robot_go_to_reset(self):
        raise NotImplementedError("Wire PiperEnv._robot_go_to_reset.")

    def _robot_read_cameras(self):
        raise NotImplementedError("Wire PiperEnv._robot_read_cameras.")

    def _robot_read_state(self):
        raise NotImplementedError("Wire PiperEnv._robot_read_state.")

    def _robot_send_action(self, action):
        raise NotImplementedError("Wire PiperEnv._robot_send_action.")


def _validate_observation(obs: dict) -> None:
    if "images" not in obs or "state" not in obs:
        raise KeyError(f"Backend get_observation() must return {{images, state}}, got {obs.keys()}")
    for name in CAMERA_NAMES:
        if name not in obs["images"]:
            raise KeyError(f"Camera '{name}' missing; got {list(obs['images'])}")
    state = np.asarray(obs["state"])
    if state.shape != (STATE_DIM,):
        raise ValueError(f"state must be (14,), got {state.shape}")
