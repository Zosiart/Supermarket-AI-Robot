from QLearning import QLearning
import numpy as np

class MyQLearning(QLearning):
    def update_q(self, state, action, r, state_next, possible_actions, alfa, gamma):
        # TODO Auto-generated method stub
        # q(s,a)_new = q(s,a)_old + alfa * (r + gamma * q(s',a_max) - q(s,a)_old)

        q_old = self.get_q(state, action)
        next_best = np.max(self.get_action_values(state_next, possible_actions))
        q_new = q_old + alfa * (r + gamma * next_best - q_old)
        self.set_q(state, action, q_new)

