import time
from Maze import Maze
from PathSpecification import PathSpecification

# Class representing the first assignment. Finds the shortest path between two points in a maze according to a specific
# path specification.
from Ant import Ant
from Route import Route


class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation, convergence):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation
        self.convergence = convergence

    # Loop that starts the shortest path process
    # @param spec Spefication of the route we wish to optimize
    # @return ACO optimized route
    def find_shortest_route(self, path_specification):
        self.maze.reset()
        return self.ant_colony(path_specification)

    def ant_colony(self, path_specification):
        best_route = Route(path_specification.start)
        # set initial value to max int
        min_size = 9223372036854775807
        counter = 0
        for iteration in range(self.generations):

            # routes taken by ants
            ant_routes = []

            # find routes of all ants
            for ant_index in range(self.ants_per_gen):
                ant = Ant(self.maze, path_specification)
                ant_routes.append(ant.find_route())

            # find the best route
            for route in ant_routes:
                if route.size() < min_size:
                    min_size = route.size()
                    best_route = route
                    counter = 0
                else:
                    counter += 1

            # if we have reached convergence, stop
            if counter >= self.convergence:
                return best_route

            # decrease value of pheromones on each link by given evaporation rate
            self.maze.evaporate(self.evaporation)
            # update pheromone value on each link which was used by ants
            self.maze.add_pheromone_routes(ant_routes, self.q)


        return best_route
