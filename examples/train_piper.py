#! /usr/bin/env python
"""DSRL online training entrypoint for the AgileX (PiperX) bimanual arms.

Steers a frozen pi0.5 policy (config ``pi05_piperx_flatten``, served remotely by
piperx-openpi with noise support) using SAC in the 32-D flow-matching noise
space. SAC observes the 3 cameras + 14-D proprio; the diffusion policy is never
updated.
"""

import os
import tempfile
import logging
from functools import partial

import numpy as np
import jax
import gymnasium as gym
from gym.spaces import Dict, Box
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from examples.train_utils_piper import trajwise_alternating_training_loop
from examples.openpi_ws_client import PiperWebsocketClientPolicy
from examples.piper_env import PiperEnv

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """Defines the SAC observation / action spaces for the PiperX setup."""

    def __init__(self, variant):
        self.variant = variant
        # 3 cameras concatenated channel-wise: (R, R, 3*num_cameras, 1)
        self.image_shape = (variant.resize_image, variant.resize_image,
                            3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            # 14-D proprio (6 joints + gripper per arm). No pi0 prefix embedding.
            state_dim = 14
            obs_dict['state'] = Box(low=-np.inf, high=np.inf, shape=(state_dim, 1),
                                    dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        # 32 is the per-step flow-matching noise dim of pi0.5.
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    logging.info('num devices %d', num_devices)
    logging.info('batch size %d', variant.batch_size)
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project,
                               experiment_id=expname, output_dir=wandb_output_dir,
                               group_name=group_name)

    # Remote pi0.5 policy server (piperx-openpi, patched for noise steering).
    agent_dp = PiperWebsocketClientPolicy(
        host=os.environ['remote_host'],
        port=int(os.environ['remote_port']),
    )
    server_metadata = agent_dp.get_server_metadata()
    logging.info(f"Server metadata: {server_metadata}")

    # The pi05_piperx_flatten config ships reset_pose in policy_metadata; prefer
    # it so RL episodes start from the same pose used during data collection.
    reset_pose = None
    if isinstance(server_metadata, dict):
        reset_pose = server_metadata.get('reset_pose', variant.get('reset_pose', None))

    logging.info("initializing PiperX environment...")
    env = PiperEnv(control_hz=variant.control_hz, reset_pose=reset_pose)
    eval_env = env
    logging.info("created the PiperX env!")

    robot_config = dict(
        camera_to_use='cam_front',
        max_timesteps=variant.max_timesteps,
    )

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    logging.info('sample obs shapes %s', [(k, v.shape) for k, v in sample_obs.items()])
    logging.info('sample action shape %s', sample_action.shape)

    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    print('Warming up SAC (JAX compile)...', flush=True)
    agent.sample_actions(sample_obs)
    print('SAC warmup done.', flush=True)

    if variant.restore_path != '':
        logging.info('restoring from %s', variant.restore_path)
        agent.restore_checkpoint(variant.restore_path)

    online_buffer_size = 2 * variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space,
                                        dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer,
                                       replay_buffer, wandb_logger, shard_fn=shard_fn,
                                       agent_dp=agent_dp, robot_config=robot_config)
