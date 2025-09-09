import numpy as np

class Maze:
    def __init__(self, size=(5,5), start=(0,0), goal=(4,4), walls=None):
        self.size = size
        self.start = start
        self.goal = goal
        self.walls = walls if walls else [(1,1), (2,2), (3,1)]
        self.reset()

    def reset(self):
        self.agent_pos = list(self.start)
        return tuple(self.agent_pos)

    def step(self, action):
        moves = {
            0: (-1, 0), # up
            1: (1, 0),  # down
            2: (0, -1), # left
            3: (0, 1)   # right
        }
        move = moves[action]
        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]

        # sınırlar içinde mi?
        if (0 <= new_pos[0] < self.size[0]) and (0 <= new_pos[1] < self.size[1]):
            if tuple(new_pos) not in self.walls:
                self.agent_pos = new_pos

        reward = -1
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 10
            done = True

        return tuple(self.agent_pos), reward, done
