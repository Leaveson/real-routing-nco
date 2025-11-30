import argparse
import os
import time
import warnings

import torch

from rl4co.data.dataset import TensorDictDataset
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.ops import batchify, unbatchify
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rrnco.baselines.routefinder.model import RouteFinderBase
from rrnco.baselines.AAFM.model import AAFM
from rrnco.envs.atsp import ATSPEnv
from rrnco.envs.rcvrp import RCVRPEnv
from rrnco.envs.rmtvrp import RMTVRPEnv
from rrnco.envs.smtvrp import SMTVRPEnv
from rrnco.models import RRNet
from rrnco.models.utils.transforms import StateAugmentation

augment = StateAugmentation(augment_fn="dihedral8", no_aug_coords=False)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# no rankzero warning
warnings.filterwarnings("ignore", category=UserWarning)


# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass
torch.set_float32_matmul_precision("medium")


def get_dataloader(td, batch_size=4):
    """Get a dataloader from a TensorDictDataset"""
    # Set up the dataloader
    dataloader = DataLoader(
        TensorDictDataset(td.clone()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TensorDictDataset.collate_fn,
    )
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem", type=str, default="atsp", help="Problem name: hcvrp, omdcpdp, etc."
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument(
        "--decode_type",
        type=str,
        default="greedy",
        help="Decoding type. Available only: greedy",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")
    parser.add_argument("--problem_size", type=int, default=100)
    # Use load_from_checkpoint with map_location, which is handled internally by Lightning
    # Suppress FutureWarnings related to torch.load and weights_only
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()
    generator_params = {"num_loc": opts.problem_size}
    batch_size = opts.batch_size
    decode_type = opts.decode_type
    checkpoint_path = opts.checkpoint
    problem = opts.problem
    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if checkpoint_path is None:
        assert (
            problem is not None
        ), "Problem must be specified if checkpoint is not provided"
        checkpoint_path = f"./checkpoints/rrnet/{problem}/epoch_199.ckpt"
    if opts.datasets is None:
        assert problem is not None, "Problem must be specified if dataset is not provided"
        data_paths = [f"./data/{problem}/{f}" for f in os.listdir(f"./data/{problem}")]
    else:
        data_paths = [opts.datasets] if isinstance(opts.datasets, str) else opts.datasets
    data_paths = sorted(data_paths)  # Sort for consistency

    if opts.no_aug:
        n_aug = 1
    else:
        n_aug = 8
    if problem == "atsp" or problem == "rcvrptw" or "routefinder" in checkpoint_path:
        n_start = 100
    else:
        if "aafm" in checkpoint_path:
            n_start = 100
        else:
            n_start = 101
    # Load the checkpoint as usual
    print("Loading checkpoint from ", checkpoint_path)

    # monkey patch for RF-based models
    if "routefinder" in checkpoint_path:
        RRNet = RouteFinderBase
    if "aafm" in checkpoint_path:
        RRNet = AAFM
    # model = RRNet(env=RMTVRPEnv())
    model = RRNet.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=False, load_baseline=False
    )
    policy = model.policy.to(device).eval()  # Use mixed precision if supported
    calc_reward = False

    for dataset in data_paths:
        costs = []
        inference_times = []

        print(f"Loading {dataset}")

        td_test = load_npz_to_tensordict(dataset)

        match problem:
            case "atsp":
                env = ATSPEnv(check_solution=False, generator_params=generator_params)
            case "rcvrptw":
                env = RMTVRPEnv(check_solution=False, generator_params=generator_params)
            case "rcvrp":
                if "routefinder" in checkpoint_path:
                    td_test["demand_linehaul"] = td_test["demand"] / td_test[
                        "capacity"
                    ].unsqueeze(-1)
                    td_test["capacity"] = torch.ones_like(td_test["capacity"])
                    td_test["locs"] = torch.cat(
                        [td_test["depot"][..., None, :], td_test["locs"]], dim=-2
                    )
                    env = RMTVRPEnv(
                        check_solution=False, generator_params=generator_params
                    )
                else:
                    td_test["demand"] = td_test["demand"] / td_test["capacity"].unsqueeze(
                        -1
                    )
                    td_test["capacity"] = torch.ones_like(td_test["capacity"])
                    env = RCVRPEnv(
                        check_solution=False, generator_params=generator_params
                    )
            case "smtvrp":
                env = SMTVRPEnv(check_solution=False, generator_params=generator_params)
                # env = SMTVRPEnv(check_solution=False, generator_params=generator_params, seed=1234, device=device)
            case _:
                raise ValueError(f"Problem {problem} not supported")
        
        dataloader = get_dataloader(td_test, batch_size=batch_size)
        with (
            torch.autocast("cuda") if "cuda" in opts.device else torch.inference_mode()
        ):  # Use mixed precision if supported
            with torch.inference_mode():
                # 각 instance별 reward를 저장할 리스트 (robustness 측정용)
                instance_rewards_all = []  # shape: (num_iterations, num_instances)
                
                for i in range(5):
                    print(f"Iteration {i}")
                    iteration_rewards = []  # 현재 iteration의 모든 instance별 reward
                    
                    for td_test_batch in tqdm(dataloader):
                        if not opts.no_aug:
                            td_test_batch = augment(td_test_batch)
                        
                        td_reset = env.reset(td_test_batch).to(device)
                        n_start = env.get_num_starts(td_reset)
                        
                        start_time = time.time()
                        if problem == "smtvrp":
                            calc_reward = True
                        out = policy(
                            td_reset,
                            env,
                            num_starts=n_start,
                            return_actions=True,
                            phase="val",
                            calc_reward=calc_reward
                        )
                        td_batch = batchify(
                            td_reset, n_start
                        )  # Expand td to batch_size * num_starts to calc. reward
                        if env.normalize:
                            real_r, norm_r = env.get_reward(td_batch, out["actions"])
                            reward = real_r
                        else:
                            if problem == "smtvrp":
                                reward = out["reward"]
                                # INSERT_YOUR_CODE
                                # out["actions"]: (batch, seq_len), count (per batch) non-overlapping zero runs

                                # import numpy as np
                                # actions_np = out["actions"].cpu().numpy()
                                # nonoverlap_zero_runs = []
                                # for row in actions_np:
                                #     is_zero = row == 0
                                #     shifted = np.r_[False, is_zero[:-1]]
                                #     zero_run_starts = (is_zero & ~shifted).sum()
                                #     if len(nonoverlap_zero_runs) == 0:
                                #         max_zero_run = zero_run_starts
                                #     else:
                                #         max_zero_run = max(max_zero_run, zero_run_starts)
                            else:
                                reward = env.get_reward(td_batch, out["actions"])
                        end_time = time.time()
                        inference_time = end_time - start_time
                        max_reward = (
                            unbatchify(reward, (n_aug, n_start)).max(dim=-1)[0].max(dim=-1)[0]
                        )
                        # 각 instance별 max_reward 저장 (batch 내 모든 instance)
                        iteration_rewards.append(max_reward)
                        
                        costs.append(max_reward.mean().item())
                        inference_times.append(inference_time)
                    
                    # 현재 iteration의 모든 instance reward를 하나로 합침
                    iteration_rewards = torch.cat(iteration_rewards, dim=0)  # (total_instances,)
                    instance_rewards_all.append(iteration_rewards)

                # instance_rewards를 (num_iterations, num_instances) 형태로 stack
                instance_rewards_all = torch.stack(instance_rewards_all, dim=0)  # (5, num_instances)
                
                # smtvrp일 때만 robustness 분석 수행
                if problem == "smtvrp":
                    num_evals = instance_rewards_all.shape[0]
                    num_instances = instance_rewards_all.shape[1]
                    
                    # 각 instance별 variance 계산: (x - mean_x)**2 / n
                    instance_mean = instance_rewards_all.mean(dim=0)  # (num_instances,)
                    instance_variances = ((instance_rewards_all - instance_mean.unsqueeze(0)) ** 2).sum(dim=0) / num_evals  # (num_instances,)
                    
                    # 데이터셋 이름 추출
                    dataset_name = os.path.basename(dataset).replace('.npz', '')
                    
                    # 테이블 형식으로 출력
                    print("\n" + "=" * 120)
                    print(f"Dataset: {dataset_name} - Instance별 max_aug_reward (각 Evaluation)")
                    print("=" * 120)
                    
                    # 헤더 출력
                    header = f"{'Instance':<10}"
                    for eval_idx in range(num_evals):
                        header += f"{'Eval ' + str(eval_idx + 1):<16}"
                    header += f"{'Mean':<16}{'Variance':<16}"
                    print(header)
                    print("-" * 120)
                    
                    # 각 instance별 데이터 출력
                    rewards_np = instance_rewards_all.cpu().numpy()
                    means_np = instance_mean.cpu().numpy()
                    variances_np = instance_variances.cpu().numpy()
                    
                    for inst_idx in range(num_instances):
                        row = f"{inst_idx:<10}"
                        for eval_idx in range(num_evals):
                            row += f"{rewards_np[eval_idx, inst_idx]:<16.6f}"
                        row += f"{means_np[inst_idx]:<16.6f}{variances_np[inst_idx]:<16.10f}"
                        print(row)
                    
                    # 요약 통계
                    print("-" * 120)
                    print(f"\n=== Robustness Summary ({num_evals} evaluations, {num_instances} instances) ===")
                    print(f"Per-instance variance (mean): {instance_variances.mean().item():.10f}")
                    print(f"Per-instance variance (std):  {instance_variances.std().item():.10f}")
                    print(f"Per-instance variance (min):  {instance_variances.min().item():.10f}")
                    print(f"Per-instance variance (max):  {instance_variances.max().item():.10f}")

                print(f"\nAverage cost:\n{-sum(costs)/len(costs):.4f}")
                print(
                    f"Per step inference time (s):\n{sum(inference_times)/len(inference_times):.4f}"
                )
                print(f"Total inference time (s):\n{sum(inference_times):.4f}")
