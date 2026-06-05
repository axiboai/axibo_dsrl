import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.train_piper import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10, help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='piperx', help='name of environment')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the observations', type=int)
    parser.add_argument('--wandb_project', default='dsrl_piperx', help='wandb project')
    parser.add_argument('--num_initial_traj_collect', default=1, help='number of trajectories to collect before starting online updates', type=int)
    parser.add_argument('--bc_rollout_episodes', default=5,
                        help='episodes of small-random-shift bootstrap before SAC steers the shift', type=int)
    parser.add_argument('--bootstrap_shift_std', default=0.3,
                        help='std of the random shift during bootstrap (gives SAC action diversity; keep small to stay near BC)', type=float)
    parser.add_argument('--steer_noise_clip', default=0.5,
                        help='clip the 32-D noise shift to +/- this before adding to the base N(0,1) noise (safety bound)', type=float)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of gradient steps per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=128, help='the size of image for the SAC encoder', type=int)
    parser.add_argument('--query_freq', default=25,
                        help='control steps per policy query (open-loop chunk consume; <= action_horizon)', type=int)
    parser.add_argument('--action_horizon', default=50, help='pi0.5 action horizon / noise length', type=int)
    parser.add_argument('--control_hz', default=30, help='robot control loop rate (Hz)', type=int)
    parser.add_argument('--max_timesteps', default=300, help='max timesteps per episode', type=int)
    parser.add_argument('--instruction', default='pick towel from pile, fold and stack', help='language instruction')
    parser.add_argument('--restore_path', default='', help='path to restore a SAC checkpoint from')

    # SAC hyperparameters (match the Franka real defaults; tuned for real-robot).
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(1024, 1024, 1024),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(3, 2, 2, 2),
        cnn_padding='VALID',
        latent_dim=50,
        discount=0.99,
        tau=0.005,
        critic_reduction='min',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy=0.0,
        num_qs=2,
        action_magnitude=2.5,
        num_cameras=3,
    )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
