import random
import numpy as np
from agents.BaseAgent import BaseAgent

class SimpleAgent(BaseAgent):
    """
    This agent tries to learn haha
    """

    def __init__(self, num_of_actions: int, epsilon: float):
        self.num_of_actions = num_of_actions
        self.Q = np.zeros(num_of_actions)
        self.N = np.zeros(num_of_actions)
        self.epsilon = epsilon

    def get_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.num_of_actions)
        else:
            return int(self.Q.argmax())

    def learn(self, action, reward) -> None:
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action])/self.N[action]
