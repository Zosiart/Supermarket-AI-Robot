import random

import numpy as np

from Route import Route

#Class that represents the ants functionality.
from Direction import Direction


class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    def __init__(self, maze, path_specification):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.rand = np.random
        # self.maxIterations = 10
        # self.antNumber = 10
        # self.pheromoneDropped = 100
        # self.evaporation = 0.1
        # self.alpha = 1.0
        # self.beta = 2.0
        # self.convergence = 10



    # find all possible directions the ant can go to, with corresponding pheromone value;
    # we return 3-tuple where [coordinates of candidate node, corresponding pheromone value,
    # direction value (should be enum)]
    def find_allowed_directions_with_its_pheromone(self, current_coordinate,seen):
        x = current_coordinate.get_x()
        y = current_coordinate.get_y()

        allowed = []
        # check west
        if x > 0 and self.maze.walls[x-1][y] == 1:
            node = current_coordinate.add_direction(2)
            if node not in seen:
                tuple = [node, self.maze.pheromones[x][y].get(2), 2]
                allowed.append(tuple)
        # check east
        if x < self.maze.width - 1 and self.maze.walls[x+1][y] == 1:
            node = current_coordinate.add_direction(0)
            if node not in seen:
                tuple = [node, self.maze.pheromones[x][y].get(0), 0]
                allowed.append(tuple)
        # check north
        if y > 0 and self.maze.walls[x][y-1] == 1:
            node = current_coordinate.add_direction(1)
            if node not in seen:
                tuple = [node, self.maze.pheromones[x][y].get(1), 1]
                allowed.append(tuple)
        # check south
        if y < self.maze.length - 1 and self.maze.walls[x][y+1] == 1:
            node = current_coordinate.add_direction(3)
            if node not in seen:
                tuple = [node, self.maze.pheromones[x][y].get(3), 3]
                allowed.append(tuple)

        return allowed

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze.
    def find_route(self):

        route = Route(self.start)
        # nodes (coordinates) seen by specific ant
        seen_nodes = []

        # search until you reach the goal
        while self.current_position != self.end:
            seen_nodes.append(self.current_position)

            #find all allowed directions
            allowed_directions = self.find_allowed_directions_with_its_pheromone(self.current_position, seen_nodes)

            # backtrack in case we reach dead end
            if len(allowed_directions) == 0:
                #print(self.current_position)
                # if we reached dead end and cannot backtrack anymore then we return empty route
                if (route.size() == 0):
                    return Route(self.start)

                last = route.remove_last()
                self.current_position = self.current_position.subtract_direction(last)
                continue

            # sum pheromone of possible directions
            total_pheromone = 0
            for pheromone in allowed_directions:
                total_pheromone += pheromone[1]

            random = self.rand.random()

            current_sum = 0
            next_node = -1

            # choose direction based on probability (calculated based on the pheromone value of specific node)
            # for direction in allowed_directions:
            #     current_sum += direction[1]
            #     if random < current_sum / total_pheromone:
            #         next_node = direction[2]
            #         break
            probabilities = [p / total_pheromone for p in [direction[1] for direction in allowed_directions]]
            next_node = np.random.choice([direction[2] for direction in allowed_directions], p=probabilities)

            if next_node == -1:
                print("error no node has been chosen")

            # shorter way to get random direction, should be implemented later
            # probabilities = [p / sum(probabilities) for p in probabilities]
            # next_city = np.random.choice(allowed_cities, p=probabilities)

            # Move ant to next node (coordinate)
            route.add(next_node)
            self.current_position = self.current_position.add_direction(next_node)

        return route
