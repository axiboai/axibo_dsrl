"""DSRL rollout + online-training utilities for the AgileX (PiperX) bimanual arms.

Mirrors ``train_utils_real.py`` (Franka/DROID) but swaps in the PiperX wire
format expected by the ``pi05_piperx_flatten`` policy server:

  * 3 cameras: cam_front, cam_left_wrist, cam_right_wrist (CHW uint8 for the
    policy; concatenated HWC for the SAC pixel encoder).
  * 14-D proprio state (no pi0 prefix embedding -- SAC state is raw qpos).
  * 14-D actions forwarded to the robot unchanged (no gripper binarization /
    no clipping).

The diffusion policy (pi0.5) is frozen; SAC is learned ONLINE in a 32-D
flow-matching noise SHIFT space.  Each query draws fresh independent N(0, 1)
noise of shape (action_horizon, 32) -- the same in-distribution noise the model
was trained with -- and SAC adds a single broadcast 32-D shift on top.  Rewards
are sparse -1/0 from manual success labels.
"""

import os
import sys
import time
import select
import tty
import termios

import numpy as np
import jax
from tqdm import tqdm
from openpi_client import image_tools
from moviepy.editor import ImageSequenceClip

from examples.piper_env import CAMERA_NAMES, CAM_FRONT


# ---------------------------------------------------------------------------#
# Observation helpers
# ---------------------------------------------------------------------------#
def _extract_observation(obs_dict):
    """Normalize a PiperEnv observation into a flat dict of arrays.

    Returns RGB uint8 HxWx3 images keyed by camera name plus the 14-D state.
    """
    images = obs_dict["images"]
    out = {"state": np.asarray(obs_dict["state"], dtype=np.float32)}
    for name in CAMERA_NAMES:
        img = np.asarray(images[name])
        # Drop alpha if present, ensure 3-channel uint8.
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        out[name] = img
    return out


def get_pi05_input(obs, instruction):
    """Build the request dict the pi05_piperx_flatten server expects.

    Matches ``make_piperx_example()`` in piperx-openpi: CHW uint8 images under
    the PiperX camera names, raw 14-D state, optional prompt.
    """

    def prep(img_hwc):
        u8 = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(img_hwc), 224, 224)
        )
        return np.transpose(u8, (2, 0, 1))  # HWC -> CHW

    return {
        "state": np.asarray(obs["state"], dtype=np.float32),
        "images": {name: prep(obs[name]) for name in CAMERA_NAMES},
        "prompt": instruction,
    }


def build_noise(mode, rng_key, vec, action_horizon, action_dim=32, *, clip=None):
    """Build the ``(1, action_horizon, action_dim)`` flow noise to send to pi0.5.

    Returns ``(noise, stored_vec)`` where ``noise`` is sent to the policy server
    and ``stored_vec`` (32-D) is the SAC action recorded in the replay buffer.

    mode='tiled' (paper-faithful):
        The 32-D ``vec`` IS the noise, repeated across every horizon step.  The
        SAC action fully determines the diffusion output (clean credit
        assignment), matching the original DSRL real/sim code.

    mode='shift':
        ``vec`` is a mean shift added on top of fresh independent ``N(0, 1)``
        base noise per step.  In-distribution and gentle, but SAC only controls
        the mean, so credit assignment is noisier.
    """
    vec = np.asarray(vec, dtype=np.float32).reshape(action_dim)
    if clip is not None:
        vec = np.clip(vec, -clip, clip)
    if mode == 'tiled':
        noise = np.repeat(vec[None, None, :], action_horizon, axis=1)
    elif mode == 'shift':
        base = np.asarray(
            jax.random.normal(rng_key, (1, action_horizon, action_dim)), dtype=np.float32)
        noise = base + vec[None, None, :]
    else:
        raise ValueError(f"unknown noise_mode: {mode!r} (expected 'tiled' or 'shift')")
    return noise, vec


def infer_pi05_actions(agent_dp, request_data, *, noise=None):
    """Call the pi0.5 policy server exactly like ``smoke_bc_piper.py``."""
    if noise is None:
        actions = agent_dp.infer(request_data)["actions"]
    else:
        actions = agent_dp.infer(request_data, noise=np.asarray(noise, dtype=np.float32))["actions"]
    actions = np.asarray(actions)
    if actions.ndim == 3:
        actions = actions[0]
    return actions


def process_images(variant, obs):
    """Concatenate the 3 cameras into the SAC pixel encoder input.

    Output shape: (1, R, R, 3*num_cameras, 1) to match DummyEnv.image_shape.
    """
    R = variant.resize_image
    ims = [
        image_tools.resize_with_pad(obs[name], R, R) for name in CAMERA_NAMES
    ]
    img_all = np.concatenate(ims, axis=2)[np.newaxis, ..., np.newaxis]
    return img_all


# ---------------------------------------------------------------------------#
# Rollout / training helpers
# ---------------------------------------------------------------------------#
def _wait_for_episode_ready(episode_num: int) -> None:
    """Pause between episodes so the operator can reset the scene."""
    print(
        f"\n>>> Press ENTER (or 'c') when the robot/scene is ready for episode {episode_num}...",
        flush=True,
    )
    try:
        line = input()
    except EOFError:
        return
    if line.strip().lower() not in ("", "c"):
        print("(continuing anyway)", flush=True)


def _write_episode_video(variant, image_list, traj_id, control_hz) -> None:
    if not image_list:
        return
    print(f"Writing episode video for traj {traj_id}...", flush=True)
    video_path = os.path.join(variant.outputdir, f'video_front_{traj_id}.mp4')
    video = np.stack(image_list)
    ImageSequenceClip(list(video), fps=control_hz).write_videofile(
        video_path, codec="libx264", logger=None)


# ---------------------------------------------------------------------------#
# Online training loop
# ---------------------------------------------------------------------------#
def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer,
                                       replay_buffer, wandb_logger, shard_fn=None,
                                       agent_dp=None, robot_config=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    i = 0
    total_env_steps = 0
    total_num_traj = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            if total_num_traj > 0:
                _wait_for_episode_ready(total_num_traj + 1)

            mode = ("BC rollout" if total_num_traj < variant.bc_rollout_episodes
                    else "SAC steered")
            print(f"\n=== Starting episode {total_num_traj + 1} ({mode}) ===", flush=True)
            traj = collect_traj(variant, agent, env, i, agent_dp, wandb_logger,
                                total_num_traj, robot_config)
            total_num_traj += 1
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', total_num_traj)
            print('total env steps:', total_env_steps)

            if i == 0:
                num_gradsteps = 5000
            else:
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step
            print(f'\n>>> Training SAC: {num_gradsteps} gradient steps '
                  f'(GPU busy, arms idle — this is normal).', flush=True)

            if total_num_traj >= variant.num_initial_traj_collect:
                report_every = max(1, min(500, num_gradsteps // 10))
                for grad_idx in range(num_gradsteps):
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1

                    if (grad_idx + 1) % report_every == 0 or grad_idx + 1 == num_gradsteps:
                        print(f'    SAC training {grad_idx + 1}/{num_gradsteps} '
                              f'(global step {i})', flush=True)

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': total_num_traj}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)

                    if variant.checkpoint_interval != -1:
                        if i % variant.checkpoint_interval == 0:
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

            print(f'>>> SAC training done for episode {total_num_traj}.', flush=True)
            _write_episode_video(
                variant, traj.get('images', []), total_num_traj - 1, variant.control_hz)


def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions'])  # (T, *action_chunk_shape) -- SAC noise actions
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)

        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon,
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()


def collect_traj(variant, agent, env, i, agent_dp=None, wandb_logger=None,
                 traj_id=None, robot_config=None):
    query_frequency = variant.query_freq
    action_horizon = variant.action_horizon
    instruction = variant.instruction
    max_timesteps = robot_config['max_timesteps']
    agent._rng, rng = jax.random.split(agent._rng)

    try:
        env.reset()
    except Exception:
        print("Environment reset failed")
        import traceback
        traceback.print_exc()
        import pdb; pdb.set_trace()

    step_time = 1.0 / variant.control_hz
    last_step_time = time.time()

    rewards = []
    action_list = []
    obs_list = []
    image_list = []
    is_success = False
    action = None

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        for t in tqdm(range(max_timesteps)):
            # Allow the operator to cut the episode short with 'q'.
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                if sys.stdin.read(1).lower() == 'q':
                    print("'q' pressed, stopping loop.")
                    break

            try:
                _env_obs = env.get_observation()
            except Exception:
                print("Environment get obs failed")
                import traceback
                traceback.print_exc()
                import pdb; pdb.set_trace()

            curr_obs = _extract_observation(_env_obs)
            image_list.append(curr_obs[CAM_FRONT])
            request_data = get_pi05_input(curr_obs, instruction)

            if t % query_frequency == 0:
                rng, key = jax.random.split(rng)
                img_all = process_images(variant, curr_obs)
                qpos = curr_obs["state"]
                obs_dict = {
                    'pixels': img_all,
                    'state': qpos[np.newaxis, ..., np.newaxis],
                }

                use_bc_infer = traj_id < variant.bc_rollout_episodes
                noise_mode = variant.noise_mode

                if use_bc_infer:
                    # Bootstrap with a random action so the SAC critic gets
                    # diversity to learn Q(s, a) (a flat bootstrap makes the actor
                    # extrapolate to the boundary -> arms fling off).  Stored
                    # action == applied action either way.
                    rng, vkey = jax.random.split(rng)
                    if noise_mode == 'tiled':
                        # Paper-faithful: standard-normal 32-D, tiled (in-dist scale).
                        vec = np.asarray(
                            jax.random.normal(vkey, agent.action_chunk_shape),
                            dtype=np.float32)
                    else:
                        # Shift: small random mean shift to stay near base BC.
                        vec = variant.bootstrap_shift_std * np.asarray(
                            jax.random.normal(vkey, agent.action_chunk_shape),
                            dtype=np.float32)
                    if t == 0:
                        print(f"BC bootstrap ({noise_mode}).", flush=True)
                else:
                    if t == 0:
                        print(f"Querying SAC for steered noise ({noise_mode})...",
                              flush=True)
                    vec = np.reshape(
                        agent.sample_actions(obs_dict), agent.action_chunk_shape)

                # Clip only applies to the shift mode (tiled noise IS the action,
                # already bounded by the actor's tanh +/- action_magnitude).
                clip = variant.steer_noise_clip if noise_mode == 'shift' else None
                noise, vec_stored = build_noise(
                    noise_mode, key, vec, action_horizon, clip=clip)
                actions_noise = vec_stored.reshape(agent.action_chunk_shape)
                if t == 0:
                    print(f"  noise stats: min={noise.min():.2f} max={noise.max():.2f} "
                          f"mean={noise.mean():.2f} | action |max|={np.abs(actions_noise).max():.2f}",
                          flush=True)
                action = infer_pi05_actions(agent_dp, request_data, noise=noise)

                if t == 0:
                    print(f"action chunk shape from server: {action.shape}")

                action_list.append(actions_noise)
                obs_list.append(obs_dict)

            # actions are (action_horizon, 14) in robot units (rad + gripper m).
            action_t = np.asarray(action[t % query_frequency], dtype=np.float32).reshape(-1)
            try:
                env.step(action_t)
            except Exception:
                print("Environment step failed")
                import traceback
                traceback.print_exc()
                import pdb; pdb.set_trace()

            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        print("Trial finished. Mark as (1) Success or (0) Failure:")
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input == '1':
                    print("Trial marked as SUCCESS.")
                    is_success = True
                    break
                elif char_input == '0':
                    print("Trial marked as FAILURE.")
                    is_success = False
                    break
                else:
                    print("Invalid input. Enter '1' for Success or '0' for Failure:")
            time.sleep(0.01)

        # add the final observation
        try:
            _env_obs = env.get_observation()
        except Exception:
            print("Environment get obs failed")
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()

        curr_obs = _extract_observation(_env_obs)
        image_list.append(curr_obs[CAM_FRONT])
        img_all = process_images(variant, curr_obs)
        qpos = curr_obs["state"]
        obs_list.append({
            'pixels': img_all,
            'state': qpos[np.newaxis, ..., np.newaxis],
        })
        print('Rollout Done')

    finally:
        query_steps = len(action_list)
        if is_success:
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)

        if wandb_logger is not None:
            wandb_logger.log({'is_success': int(is_success)}, step=i)
            wandb_logger.log({'total_num_traj': traj_id}, step=i)

        print("Episode finished — moving to reset pose.", flush=True)
        try:
            env.reset()
        except Exception:
            print("Environment reset failed")
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    traj = {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'env_steps': len(action_list),
        'images': image_list,
    }
    return traj
