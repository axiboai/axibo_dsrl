"""PiperEnv backend wired to the piperx_lerobot_setup ZMQ stack.

Mirrors openpi_inference.py I/O (3335/5560/5556/5558 in, 3336 out) without
running the policy loop — DSRL calls the policy server with noise steering.

Do NOT run openpi_inference.py at the same time (both publish on 3336).

Pre-flight:
    follower_sink.py, RealSense publishers, teleop on 3335 (--no-command),
    policy server on :8000 (DSRL client, not this module).

Usage:
    export PIPER_LEROBOT_SETUP=~/piperx_lerobot_setup/scripts
    export PIPER_ENV_FACTORY=examples.piper_env_zmq:make_piper_env
    python3 examples/smoke_bc_piper.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

_LEROBOT_SCRIPTS = Path(
    os.environ.get("PIPER_LEROBOT_SETUP", "~/piperx_lerobot_setup/scripts")
).expanduser()
if str(_LEROBOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SCRIPTS))

from realtime_input import (  # noqa: E402
    CameraFrame,
    LatestStore,
    RealtimeInputManager,
    decode_teleop_json,
)

TELEOP_STATE_ADDR = os.environ.get("PIPER_STATE_ADDR", "tcp://localhost:3335")
CAM_FRONT_ADDR = os.environ.get("PIPER_CAM_FRONT_ADDR", "tcp://localhost:5560")
CAM_LEFT_ADDR = os.environ.get("PIPER_CAM_LEFT_ADDR", "tcp://localhost:5556")
CAM_RIGHT_ADDR = os.environ.get("PIPER_CAM_RIGHT_ADDR", "tcp://localhost:5558")
TARGET_PUB_ADDR = os.environ.get("PIPER_TARGET_ADDR", "tcp://0.0.0.0:3336")

TELEOP_STATE_KEY = "teleop_state"
CAM_FRONT_KEY = "cam_front"
CAM_LEFT_KEY = "cam_left_wrist"
CAM_RIGHT_KEY = "cam_right_wrist"


def pack_state_14(state_msg: dict) -> np.ndarray:
    fL = state_msg["follower_left"]
    fR = state_msg["follower_right"]
    out = np.zeros(14, dtype=np.float32)
    out[:6] = fL["q"][:6]
    out[6] = fL["gripper"]
    out[7:13] = fR["q"][:6]
    out[13] = fR["gripper"]
    return out


def _get_teleop_state(latest: LatestStore) -> Optional[dict]:
    item = latest.get(TELEOP_STATE_KEY)
    if item is None or not isinstance(item.value, dict):
        return None
    return item.value


def _get_camera_rgb(latest: LatestStore, name: str) -> Optional[np.ndarray]:
    item = latest.get(name)
    if item is None or not isinstance(item.value, CameraFrame):
        return None
    return item.value.rgb


def _resolve_cam_front(cam_front, cam_left, cam_right, *, wrist_only, front_from):
    if not wrist_only:
        return cam_front
    if front_from == "none":
        return None
    if front_from == "right":
        return cam_right
    return cam_left


def make_target_msg(seq: int, action_14: np.ndarray) -> str:
    action_14 = np.asarray(action_14, dtype=np.float32)
    return json.dumps({
        "t_mono": time.monotonic(),
        "seq": seq,
        "source": "dsrl",
        "left": {"q": action_14[0:6].tolist(), "gripper": float(action_14[6])},
        "right": {"q": action_14[7:13].tolist(), "gripper": float(action_14[13])},
    })


class PiperZmqEnv:
    def __init__(self, reset_pose=None, *, wrist_only=False, front_from="left", warmup_timeout_s=5.0,
                 control_hz=30, reset_settle_s=2.0):
        self.reset_pose = reset_pose
        self.wrist_only = wrist_only or os.environ.get("PIPER_WRIST_ONLY", "") == "1"
        self.front_from = os.environ.get("PIPER_FRONT_FROM", front_from)
        self.control_hz = float(os.environ.get("PIPER_CONTROL_HZ", control_hz))
        self.reset_settle_s = float(os.environ.get("PIPER_RESET_SETTLE_S", reset_settle_s))
        self._seq = 0

        ctx = zmq.Context.instance()
        self._inputs = RealtimeInputManager(ctx)
        self._inputs.add_text_subscriber(TELEOP_STATE_KEY, TELEOP_STATE_ADDR, decode_teleop_json)
        if not self.wrist_only:
            self._inputs.add_camera_subscriber(CAM_FRONT_KEY, CAM_FRONT_ADDR)
        self._inputs.add_camera_subscriber(CAM_LEFT_KEY, CAM_LEFT_ADDR)
        self._inputs.add_camera_subscriber(CAM_RIGHT_KEY, CAM_RIGHT_ADDR)
        self._inputs.start()

        self._pub = ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.bind(TARGET_PUB_ADDR)
        print(f"[PiperZmqEnv] target PUB -> {TARGET_PUB_ADDR}", flush=True)
        self._warmup(warmup_timeout_s)

    @property
    def _latest(self) -> LatestStore:
        return self._inputs.latest

    def _warmup(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = _get_teleop_state(self._latest)
            if state is not None:
                if self.wrist_only:
                    ok = (
                        _get_camera_rgb(self._latest, CAM_LEFT_KEY) is not None
                        and _get_camera_rgb(self._latest, CAM_RIGHT_KEY) is not None
                    )
                else:
                    ok = all(
                        _get_camera_rgb(self._latest, k) is not None
                        for k in (CAM_FRONT_KEY, CAM_LEFT_KEY, CAM_RIGHT_KEY)
                    )
                if ok:
                    print("[PiperZmqEnv] ZMQ warmup OK", flush=True)
                    return
            time.sleep(0.05)
        print("[PiperZmqEnv] WARNING: warmup timed out", flush=True)

    def reset(self):
        if self.reset_pose is not None and len(self.reset_pose) >= 6:
            home = np.asarray(self.reset_pose[:6], dtype=np.float32)

            # Preserve current gripper openings (home pose is joints only).
            grip_l, grip_r = 0.0, 0.0
            state = _get_teleop_state(self._latest)
            if state is not None:
                try:
                    grip_l = float(state["follower_left"]["gripper"])
                    grip_r = float(state["follower_right"]["gripper"])
                except (KeyError, TypeError, IndexError):
                    pass

            # Drive BOTH arms to the home joints.
            action = np.zeros(14, dtype=np.float32)
            action[:6] = home
            action[6] = grip_l
            action[7:13] = home
            action[13] = grip_r

            # Command the home target repeatedly so both arms actually arrive
            # (a single publish does not move the follower all the way home).
            n_steps = max(1, int(self.reset_settle_s * self.control_hz))
            dt = 1.0 / self.control_hz
            print(f"[PiperZmqEnv] resetting both arms to home for {self.reset_settle_s:.1f}s...",
                  flush=True)
            for _ in range(n_steps):
                self.step(action)
                time.sleep(dt)
        time.sleep(0.5)
        return self.get_observation()

    def get_observation(self) -> dict:
        state_msg = _get_teleop_state(self._latest)
        if state_msg is None:
            raise RuntimeError("No teleop state on ZMQ (3335). Is teleop publisher running?")
        latest = self._latest
        cam_front = _get_camera_rgb(latest, CAM_FRONT_KEY) if not self.wrist_only else None
        cam_left = _get_camera_rgb(latest, CAM_LEFT_KEY)
        cam_right = _get_camera_rgb(latest, CAM_RIGHT_KEY)
        front = _resolve_cam_front(
            cam_front, cam_left, cam_right,
            wrist_only=self.wrist_only, front_from=self.front_from,
        )

        def _hwc_or_black(rgb):
            if rgb is None:
                return np.zeros((224, 224, 3), dtype=np.uint8)
            img = np.asarray(rgb)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.shape[-1] == 4:
                img = img[..., :3]
            return img

        return {
            "images": {
                "cam_front": _hwc_or_black(front),
                "cam_left_wrist": _hwc_or_black(cam_left),
                "cam_right_wrist": _hwc_or_black(cam_right),
            },
            "state": pack_state_14(state_msg),
        }

    def step(self, action) -> None:
        msg = make_target_msg(self._seq, np.asarray(action, dtype=np.float32))
        try:
            self._pub.send_string(msg, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        self._seq += 1

    def close(self) -> None:
        try:
            self._pub.close(linger=0)
        except Exception:
            pass
        self._inputs.stop()


def make_piper_env(reset_pose=None) -> PiperZmqEnv:
    return PiperZmqEnv(reset_pose=reset_pose)
