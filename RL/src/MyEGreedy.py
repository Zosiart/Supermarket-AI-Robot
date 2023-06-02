from Maze import Maze
from Agent import Agent
import numpy as np


class MyEGreedy:
    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent, maze):
        # TODO to select an action at random in State s
        actions = maze.get_valid_actions(agent)
        chosen_action = np.random.choice(actions)
        # print("random action: " + chosen_action.id)
        return chosen_action

    def get_best_action(self, agent, maze, q_learning):
        # TODO to select the best possible action currently known in State s.
        valid_actions = maze.get_valid_actions(agent)
        chosen_action = np.random.choice([np.argmax(q_learning.get_action_values(agent.get_state(maze), valid_actions))])
        return valid_actions[chosen_action]

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        # TODO to select between random or best action selection based on epsilon.
        rand = np.random.random()
        if rand < epsilon:
            return self.get_random_action(agent, maze)
        else:
            return self.get_best_action(agent, maze, q_learning)
