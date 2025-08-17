import random
import numpy as np
from agents.BaseAgent import BaseAgent


class GradientAgent(BaseAgent):
    """
    This agent uses gradient ascent
    """

    def __init__(self, num_of_actions: int, baseline: int):
        self.num_of_actions = num_of_actions
        self.H = np.zeros(num_of_actions)
        self.Rt = 0
        self.baseline = baseline #0 si Rt es siempre 0 y 1 si Rt se recalcula todas las iteraciones
        self.alpha = 0.1
        self.probs = np.ones(num_of_actions)/num_of_actions
        self.actions = []
        self.steps = 0
        for i in range(self.num_of_actions):
            self.actions.append(i)

    def get_action(self) -> int:
        return random.choices(self.actions, weights=self.probs, k=1)[0]

    def learn(self, action, reward) -> None:
        self.steps += 1
        self.probs = np.exp(self.H) / np.sum(np.exp(self.H))
        self.Rt += (reward - self.Rt)/self.steps
        #Tip de la clase: restar C en softmax ya que no cambia el resultado
        #C = max(self.H)
        for actions in range(self.num_of_actions):
            if actions == action:
                self.H[actions] += (reward - self.Rt*self.baseline)*self.alpha*(1-self.probs[actions])
            else:
                self.H[actions] += -(reward - self.Rt*self.baseline)*self.alpha*self.probs[actions]
