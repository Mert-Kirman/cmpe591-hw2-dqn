import torch
from homework2 import Hw2Env
from train_dqn import DQNAgent

# Load saved model
agent = DQNAgent()
model_path = "runs/experiment_2_more_exploration_decay_20000/model.pt"  # Update this path to your saved model
agent.online_net.load_state_dict(torch.load(model_path))
agent.online_net.eval()  # Set the network to evaluation mode

N_ACTIONS = 8
env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
for episode in range(100):
    env.reset()
    done = False
    cumulative_reward = 0.0
    while not done:
        state = env.high_level_state()
        action = agent.select_action(state, training=False)  # Use the trained policy for action selection
        state, reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated
        cumulative_reward += reward
    print(f"Episode={episode}, reward={cumulative_reward}")
