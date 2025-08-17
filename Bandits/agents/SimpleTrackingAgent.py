import random
import numpy as np
from agents.BaseAgent import BaseAgent

class SimpleTrackingAgent(BaseAgent):
    """
    This agent tries manage changing rewards
    """

    def __init__(self, num_of_actions: int, epsilon: float):
        self.num_of_actions = num_of_actions
        if epsilon == 0:
            self.Q = np.ones(num_of_actions)*5
        else:
            self.Q = np.zeros(num_of_actions)

        self.epsilon = epsilon
        self.alpha = 0.1

    def get_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.num_of_actions)
        else:
            return int(self.Q.argmax())

    def learn(self, action, reward) -> None:
        self.Q[action] += (reward - self.Q[action])*self.alpha
