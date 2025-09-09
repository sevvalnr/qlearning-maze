import matplotlib.pyplot as plt
import numpy as np
from maze import Maze
from qlearning import QLearningAgent

# Eğitilmiş ajanı yükleyelim
env = Maze()
agent = QLearningAgent(state_size=env.size, action_size=4)
agent.q_table = np.load("qtable.npy")  # train sonrası kaydedilmiş tablo

# Labirenti çiz
grid = np.zeros(env.size)
for wall in env.walls:
    grid[wall] = -1
grid[env.goal] = 2

plt.imshow(grid, cmap="coolwarm", origin="upper")

state = env.reset()
done = False

while not done:
    action = np.argmax(agent.q_table[state])
    state, _, done = env.step(action)
    plt.scatter(state[1], state[0], c="yellow")

plt.show()
