#!/bin/bash
# DSRL online fine-tuning for the AgileX (PiperX) bimanual arms.
#
# Single-machine setup (policy server + SAC both on one 24 GB 4090):
#
#   1. Start the patched piperx-openpi policy server in a SEPARATE shell, with
#      JAX preallocation OFF so it does not grab the whole GPU:
#
#        export XLA_PYTHON_CLIENT_PREALLOCATE=false
#        # if OOM, hard-partition instead:
#        #   export XLA_PYTHON_CLIENT_PREALLOCATE=true
#        #   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
#        uv run scripts/serve_policy.py policy:checkpoint \
#          --policy.config=pi05_piperx_flatten \
#          --policy.dir=/path/to/pi05_flatten_raw/14999 \
#          --default-prompt="pick towel from pile, fold and stack"
#
#      Apply the DSRL noise patch once (restart server after):
#        python3 examples/scripts/patch_openpi_websocket_noise.py ~/openpi
#   2. ZMQ stack running (same as openpi_inference.py pre-flight):
#        follower_sink, RealSense pubs, teleop on 3335 (--no-command).
#        Do NOT run openpi_inference.py while DSRL is running.

proj_name=DSRL_pi05_PiperX
device_id=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Robot I/O via piperx_lerobot_setup ZMQ (same ports as openpi_inference.py).
export PIPER_LEROBOT_SETUP="${PIPER_LEROBOT_SETUP:-$HOME/piperx_lerobot_setup/scripts}"
export PIPER_ENV_FACTORY="${PIPER_ENV_FACTORY:-examples.piper_env_zmq:make_piper_env}"

export EXP="${REPO_ROOT}/logs/$proj_name"
export CUDA_VISIBLE_DEVICES=$device_id

# Share the GPU with the local pi0.5 server: allocate on demand instead of
# preallocating 75%. If you still OOM, set PREALLOCATE=true and cap SAC with
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 (give the server ~0.55).
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# pi0.5 policy server runs locally on this same machine.
export remote_host="127.0.0.1"
export remote_port="8000"

# Horizon note: max_timesteps=6000 (~200 s @ 30 Hz) is just a SAFETY CEILING --
# press 'q' to end an episode the moment the fold is done. Typical q-ended folds
# Typical q-ended folds are ~100 s (~120 SAC transitions @ query_freq=25).
# Re-infer every 25 steps (half the 50-step chunk) to match openpi_inference /
# ActionChunkBroker. The Bellman backup uses discount^query_freq per transition.
python3 examples/launch_train_piper.py \
--algorithm pixel_sac \
--env piperx \
--prefix dsrl_pi05_piperx \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.9995 \
--seed 0 \
--max_steps 150000 \
--eval_interval 2000 \
--log_interval 100 \
--multi_grad_step 30 \
--resize_image 128 \
--action_magnitude 2.5 \
--query_freq 25 \
--action_horizon 50 \
--control_hz 30 \
--max_timesteps 6000 \
# noise_mode: 'shift' (gentle, current) or 'tiled' (paper-faithful, clean credit).
# For 'tiled' consider lowering --action_magnitude (e.g. 1.0) since the action is
# the noise itself, held constant across the 50-step horizon.
--noise_mode shift \
--bc_rollout_episodes 5 \
--bootstrap_shift_std 0.3 \
--steer_noise_clip 0.5 \
--hidden_dims 1024 \
--num_qs 2 \
--instruction "pick towel from pile, fold and stack"
