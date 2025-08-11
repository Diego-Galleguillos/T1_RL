import random
import numpy as np
from agents.BaseAgent import BaseAgent


class GradientAgent(BaseAgent):
    """
    This agent uses gradient ascent
    """

    def __init__(self, num_of_actions: int):
        self.num_of_actions = num_of_actions
        self.H = np.zeros(num_of_actions)
        self.baseline = 4
        self.alpha = 0.1
        self.probs = np.ones(num_of_actions)/num_of_actions
        self.actions = []
        for i in range(self.num_of_actions):
            self.actions.append(i)

    def get_action(self) -> int:
        return random.choices(self.actions, weights=self.probs, k=1)[0]

    def learn(self, action, reward) -> None:
        self.probs[action] = np.exp(self.H[action]) / np.sum(np.exp(self.H))
        for actions in range(self.num_of_actions):
            if actions == action:
                self.H[actions] += (reward - self.baseline)*self.alpha*(1-self.probs[actions])
            else:
                self.H[actions] += -(reward - self.baseline)*self.alpha*self.probs[actions]
