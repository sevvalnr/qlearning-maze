from maze import Maze
from qlearning import QLearningAgent
import numpy as np

env = Maze()
agent = QLearningAgent(state_size=env.size, action_size=4)

episodes = 500

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    if (ep+1) % 50 == 0:
        print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

# ✅ Eğitim tamamlanınca Q-table kaydedelim
np.save("qtable.npy", agent.q_table)
print("Q-table saved: qtable.npy")
print("Eğitim donee.")