#! /usr/bin/env python
"""BC-only smoke test for the PiperX DSRL integration (no SAC, no replay buffer).

Validates the full client/server/robot path before launching online RL:

    * PiperEnv hooks (reset / get_observation / step) talk to piperx_lerobot_setup
    * the policy server returns 14-D actions in robot units
    * (optionally) the noise-steering protocol works end-to-end

It simply rolls out the frozen pi0.5 policy. With --gaussian-noise it exercises
the exact websocket noise path DSRL uses (server must be the patched
piperx-openpi); without it, the server samples its own noise (pure BC).

Run on the robot PC (server must already be up):

    export remote_host=<gpu-ip>
    export remote_port=8000
    python3 examples/smoke_bc_piper.py --episodes 1 --max_timesteps 200
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Repo root on sys.path (setup.py only installs jaxrl2, not examples/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from examples.openpi_ws_client import PiperWebsocketClientPolicy
from examples.piper_env import PiperEnv
from examples.train_utils_piper import (
    _extract_observation,
    get_pi05_input,
    infer_pi05_actions,
)


def rollout(agent_dp, env, args, action_horizon, action_dim=32):
    env.reset()
    step_time = 1.0 / args.control_hz
    last_step_time = time.time()
    action = None

    for t in range(args.max_timesteps):
        curr_obs = _extract_observation(env.get_observation())
        request_data = get_pi05_input(curr_obs, args.instruction)

        if t % args.query_freq == 0:
            if args.gaussian_noise:
                noise = np.random.randn(1, action_horizon, action_dim).astype(np.float32)
                action = infer_pi05_actions(agent_dp, request_data, noise=noise)
            else:
                action = infer_pi05_actions(agent_dp, request_data)
            if t == 0:
                print(f"action chunk shape from server: {action.shape}")

        action_t = np.asarray(action[t % args.query_freq], dtype=np.float32).reshape(-1)
        env.step(action_t)

        now = time.time()
        dt = now - last_step_time
        if dt < step_time:
            time.sleep(step_time - dt)
            last_step_time = time.time()
        else:
            last_step_time = now

    env.reset()
    print("rollout complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default=1, type=int)
    parser.add_argument('--max_timesteps', default=200, type=int)
    parser.add_argument('--query_freq', default=25, type=int)
    parser.add_argument('--action_horizon', default=50, type=int)
    parser.add_argument('--control_hz', default=30, type=int)
    parser.add_argument('--instruction', default='pick towel from pile, fold and stack')
    parser.add_argument('--gaussian-noise', action='store_true',
                        help='send Gaussian noise via the DSRL protocol (tests the patched server)')
    args = parser.parse_args()

    agent_dp = PiperWebsocketClientPolicy(
        host=os.environ['remote_host'],
        port=int(os.environ['remote_port']),
    )
    metadata = agent_dp.get_server_metadata()
    print(f"server metadata: {metadata}")

    reset_pose = metadata.get('reset_pose') if isinstance(metadata, dict) else None
    env = PiperEnv(control_hz=args.control_hz, reset_pose=reset_pose)

    for ep in range(args.episodes):
        print(f"=== episode {ep} ===")
        rollout(agent_dp, env, args, args.action_horizon)


if __name__ == '__main__':
    main()
