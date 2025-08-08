import random
import numpy as np
from agents.BaseAgent import BaseAgent

class SimpleAgent(BaseAgent):
    """
    This agent tries to learn haha
    """

    def __init__(self, num_of_actions: int):
        self.num_of_actions = num_of_actions
        self.Q = np.zeros(num_of_actions)
        self.current_step = 0
        self.epsilon = 0.1

    def get_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.num_of_actions)
        else:
            return int(self.Q.argmax())

    def learn(self, action, reward) -> None:
        self.current_step += 1
        self.Q[action] += (reward - self.Q[action])/self.current_step
