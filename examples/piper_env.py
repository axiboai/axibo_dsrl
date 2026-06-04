"""Minimal robot-environment adapter for the AgileX (PiperX) bimanual setup.

DSRL only needs three things from the environment:

    reset()             -> move the arms to a known start pose
    get_observation()   -> return the current cameras + proprio
    step(action)        -> command a single 14-D joint+gripper target

This file deliberately does NOT talk to the CAN bus directly. Instead it
delegates to your existing ``piperx_lerobot_setup`` stack (the same code you
already use for teleop / BC inference). Fill in the four ``_robot_*`` hooks
below and the rest of the DSRL pipeline will work unchanged.

Action / state layout (must match the LeRobot dataset + openpi config):

    index  0..5   left arm joints (rad)
    index  6      left gripper (meters, ~0 closed .. ~0.07 open)
    index  7..12  right arm joints (rad)
    index  13     right gripper (meters)

The policy server (piperx-openpi, config ``pi05_piperx_flatten``) returns
actions already in these robot units via ``PiperXOutputs``, so ``step`` should
forward them to the follower with no extra scaling or gripper binarization.
"""

import time

import numpy as np


# Camera names expected by the pi0.5 PiperX policy (NOT the LeRobot
# ``observation.images.*`` keys -- those are only used during training).
CAM_FRONT = "cam_front"
CAM_LEFT_WRIST = "cam_left_wrist"
CAM_RIGHT_WRIST = "cam_right_wrist"
CAMERA_NAMES = (CAM_FRONT, CAM_LEFT_WRIST, CAM_RIGHT_WRIST)

STATE_DIM = 14


class PiperEnv:
    """Thin wrapper around piperx_lerobot_setup for DSRL online rollouts.

    Parameters
    ----------
    control_hz:
        Loop rate used by the DSRL collect loop (dataset was recorded at 30 Hz).
    reset_pose:
        Optional per-arm reset joints. Defaults to the value stored in the
        training config's ``policy_metadata['reset_pose']``.
    robot:
        Optional pre-constructed robot handle from piperx_lerobot_setup. If
        omitted, ``_robot_connect`` is called lazily on first use.
    """

    def __init__(self, control_hz=30, reset_pose=None, robot=None):
        self.control_hz = control_hz
        self.reset_pose = reset_pose
        self._robot = robot
        if self._robot is None:
            self._robot = self._robot_connect()

    # ------------------------------------------------------------------ #
    # DSRL-facing API
    # ------------------------------------------------------------------ #
    def reset(self):
        """Return the arms to the start configuration before each episode."""
        self._robot_go_to_reset()
        # Small settle time so the first observation is not mid-motion.
        time.sleep(0.5)
        return self.get_observation()

    def get_observation(self):
        """Return a dict consumed by ``train_utils_piper._extract_observation``.

        Returns
        -------
        dict with keys:
            "images": {cam_name: HxWx3 uint8 RGB array} for all CAMERA_NAMES
            "state":  (14,) float32 joint+gripper vector in robot units
        """
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
        """Command one 14-D joint+gripper target (already in robot units)."""
        action = np.asarray(action, dtype=np.float32).reshape(STATE_DIM)
        self._robot_send_action(action)

    # ------------------------------------------------------------------ #
    # Hooks to implement against piperx_lerobot_setup  (the only TODOs)
    # ------------------------------------------------------------------ #
    def _robot_connect(self):
        """Construct and return your robot handle.

        Example (adapt to your stack):
            from piperx_lerobot_setup import BimanualPiper
            return BimanualPiper(left_can="can_left", right_can="can_right",
                                 cameras=["front", "left_wrist", "right_wrist"])
        """
        raise NotImplementedError(
            "Wire PiperEnv._robot_connect to your piperx_lerobot_setup robot handle."
        )

    def _robot_go_to_reset(self):
        """Move both arms to ``self.reset_pose`` (blocking)."""
        raise NotImplementedError(
            "Wire PiperEnv._robot_go_to_reset to your follower reset routine."
        )

    def _robot_read_cameras(self):
        """Return {CAM_FRONT/CAM_LEFT_WRIST/CAM_RIGHT_WRIST: HxWx3 uint8 RGB}."""
        raise NotImplementedError(
            "Wire PiperEnv._robot_read_cameras to your RealSense/camera reads."
        )

    def _robot_read_state(self):
        """Return the 14-D [L_joints6, L_grip, R_joints6, R_grip] vector."""
        raise NotImplementedError(
            "Wire PiperEnv._robot_read_state to your joint/gripper feedback."
        )

    def _robot_send_action(self, action):
        """Send 14-D joint+gripper targets to the followers (e.g. port 3336)."""
        raise NotImplementedError(
            "Wire PiperEnv._robot_send_action to your follower_sink command path."
        )
