import gymnasium as gym
import numpy as np
from collections import defaultdict

# Initialize Blackjack environment
env = gym.make('Blackjack-v1', render_mode='ansi')  # Use 'ansi' for text-based rendering

# Q-table
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Hyperparameters
num_episodes = 500_000
alpha = 0.1
gamma = 1.0
epsilon = 0.1

def epsilon_greedy_policy(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

# Training loop
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        action = epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        best_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
        state = next_state

# Deterministic policy (no exploration)
def policy(state):
    return np.argmax(Q[state])

# --- Playback Function ---
def play_blackjack_episode():
    state, _ = env.reset()
    done = False
    print("\n=== NEW GAME ===")
    print(env.render())
    while not done:
        action = policy(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        print(env.render())
    print(f"Game Over. Reward: {reward}\n{'='*30}")

# Play 5 sample games
for _ in range(5):
    play_blackjack_episode()
