import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-running", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=1500, help="Checkpoint iteration to load")
    parser.add_argument("--seconds", type=float, default=70.0)
    parser.add_argument("--max-speed", type=float, default=2.5)
    parser.add_argument("-o", "--output", type=str, default="go2_run.mp4")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env_cfg["episode_length_s"] = args.seconds + 10.0
    env_cfg["resampling_time_s"] = args.seconds + 10.0

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        record=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    env.cam.start_recording(save_to_filename=args.output, fps=50)
    with torch.no_grad():
        for i in range(int(args.seconds / env.dt)):
            t = i * env.dt
            env.commands[:, 0] = min(0.5 + (args.max_speed - 0.5) * max(t - 10.0, 0.0) / 30.0, args.max_speed)
            actions = policy(obs_dict)
            obs_dict, rews, dones, infos = env.step(actions)
            if i % 250 == 0:
                print(f"t={t:5.1f}s cmd={env.commands[0, 0]:.2f} vel={env.base_lin_vel[0, 0]:.2f} m/s")
    env.cam.stop_recording()


if __name__ == "__main__":
    main()

"""
# evaluation
python go2_eval.py -e go2-running --ckpt 1500
"""
