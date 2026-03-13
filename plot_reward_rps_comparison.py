import os
import numpy as np
import matplotlib.pyplot as plt

runs_to_compare = [
    "runs/baseline",
    "runs/experiment_1_lr_0.001",
    "runs/experiment_2_more_exploration_decay_20000",
    "runs/experiment_3_most_exploration_decay_30000"
]

run_results = {}

for run_dir in runs_to_compare:
    rewards_path = os.path.join(run_dir, "rewards.npy")
    rps_path = os.path.join(run_dir, "rps.npy")
    
    if os.path.exists(rewards_path) and os.path.exists(rps_path):
        rewards = np.load(rewards_path)
        rps = np.load(rps_path)
        
        run_results[run_dir] = {
            "rewards": rewards,
            "rps": rps
        }
    else:
        print(f"Metrics not found for {run_dir}")

# Plot composite moving average of rewards and RPS over episodes
plt.figure(figsize=(12, 6))
for run_dir, metrics in run_results.items():
    rewards = metrics["rewards"]
    rps = metrics["rps"]
    
    # Compute moving averages
    window_size = 100
    rewards_ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    rps_ma = np.convolve(rps, np.ones(window_size)/window_size, mode='valid')
    
    plt.subplot(2, 1, 1)
    plt.plot(rewards_ma, label=f"{run_dir.split('/')[-1]}")
    plt.title("Reward 100 MA Over Episodes")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(rps_ma, label=f"{run_dir.split('/')[-1]}")
    plt.title("RPS 100 MA Over Episodes")
    plt.xlabel('Episode')
    plt.ylabel('RPS')
    plt.legend()

plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/reward_rps_comparisons.png")
