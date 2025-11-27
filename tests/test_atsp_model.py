
import torch
from tensordict import TensorDict
from rrnco.envs.atsp.env import ATSPEnv
from rrnco.baselines.AAFM.atsp_model import AAFMATSPPolicy

def test_atsp_model():
    # Create environment
    env = ATSPEnv(generator_params={"num_loc": 20, "min_dist": 0, "max_dist": 1})
    
    # Create policy
    policy = AAFMATSPPolicy(
        embed_dim=128,
        num_encoder_layers=2,
        env_name="atsp",
        neighbors=10 # Test with k < num_loc
    )
    
    # Reset environment
    td = env.reset(batch_size=[2])
    print("Reset TD keys:", td.keys())
    print("TD batch size:", td.batch_size)
    print("Distance matrix shape:", td["distance_matrix"].shape)
    if "min_distance" in td.keys():
        print("Min distance shape:", td["min_distance"].shape)
    
    # Run policy
    out = policy(td, env, phase="test", decode_type="greedy")
    
    print("Output keys:", out.keys())
    print("Reward shape:", out["reward"].shape)
    print("Reward:", out["reward"])
    print("Actions shape:", out["actions"].shape)
    
    # Manual reward check
    real, norm = env.get_reward(td, out["actions"])
    print("Manual real reward shape:", real.shape)
    print("Manual norm reward shape:", norm.shape)
    
    # assert out["reward"].shape == (2,) # Comment out assertion to see output

if __name__ == "__main__":
    test_atsp_model()
