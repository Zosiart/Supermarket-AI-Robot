import numpy as np
import matplotlib.pyplot as plt
from TSPData import TSPData

MUTATION_PROB = 0.01

CROSSOVER_PROB = 0.7


class Candidate:
    def __init__(self):
        self.chromosome = []
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness


# TSP problem solver using genetic algorithms.
class GeneticAlgorithm:

    # Constructs a new 'genetic algorithm' object.
    # @param generations the amount of generations.
    # @param popSize the population size.
    def __init__(self, generations, pop_size):
        self.generations = generations
        self.pop_size = pop_size
        self.number_of_products = 0

    # This method should solve the TSP.
    # @param pd the TSP data.
    # @return the optimized product sequence.
    def solve_tsp(self, tsp_data):
        print("Solving TSP using genetic algorithm")
        print("Generations: " + str(self.generations))
        print("Population size: " + str(self.pop_size))
        self.number_of_products = len(tsp_data.product_locations)

        tsp_data.build_distance_lists()

        total_fitness_data = []
        average_fitness_data = []
        best_fitness_data = []
        global_best_candidate = Candidate()

        # distances_from_start = tsp_data.get_start_distances()
        # distances_to_end = tsp_data.get_end_distances()
        # distances_between_products = tsp_data.get_distances()
        # print(distances_between_products)
        # print(distances_to_end)
        # print(distances_from_start)

        # Create initial population
        population = self.create_population(tsp_data)
        print("Initial population created")

        # Run the genetic algorithm
        for i in range(self.generations):
            self.calculate_fitness_of_population(population, tsp_data)
            population.sort(reverse=True)
            if population[0].fitness > global_best_candidate.fitness:
                global_best_candidate = population[0]
            total_fitness = self.calculate_total_fitness(population)
            # print("Generation " + str(i) + " best fitness: " + str(population[0].fitness))
            # print("total fitness: " + str(total_fitness))
            total_fitness_data.append(total_fitness)
            average_fitness_data.append(total_fitness / self.pop_size)
            best_fitness_data.append(global_best_candidate.fitness)
            ranges = self.create_cumulative_ranges(population, total_fitness)
            population = self.create_new_population(population, ranges)
            if i % 10 == 0: print(i)

        print("Best chromosome: " + str(global_best_candidate.chromosome))
        plt.plot(range(self.generations), average_fitness_data)
        plt.plot(range(self.generations), best_fitness_data)
        plt.show()
        plt.plot(range(self.generations), total_fitness_data)
        plt.show()

        return global_best_candidate.chromosome

    def create_chromosome(self, tsp_data):
        chromosome = []
        for i in range(len(tsp_data.product_locations)):
            chromosome.append(i)
        np.random.shuffle(chromosome)
        return chromosome

    def create_population(self, tsp_data):
        population = []
        for i in range(self.pop_size):
            canditate = Candidate()
            canditate.chromosome = self.create_chromosome(tsp_data)
            population.append(canditate)
        return population

    def calculate_fitness(self, chromosome, distances_from_start, distances_to_end, distances_between_products):

        fitness = distances_from_start[chromosome[0]]
        for i in range(len(chromosome) - 1):
            fitness += distances_between_products[chromosome[i]][chromosome[i + 1]]
        fitness += distances_to_end[chromosome[-1]]
        return 1 / fitness

    def calculate_fitness_of_population(self, population, tsp_data):
        distances_from_start = tsp_data.get_start_distances()
        distances_to_end = tsp_data.get_end_distances()
        distances_between_products = tsp_data.get_distances()
        for candidate in population:
            candidate.fitness = self.calculate_fitness(candidate.chromosome, distances_from_start, distances_to_end,
                                                       distances_between_products)
        return population

    def calculate_total_fitness(self, population):
        total_fitness = 0
        for candidate in population:
            total_fitness += candidate.fitness
        return total_fitness

    def create_cumulative_ranges(self, population, total_fitness):
        ranges = []
        cumulative = 0
        for candidate in population:
            cumulative += candidate.fitness
            ranges.append(cumulative / total_fitness)
        return ranges

    def create_new_population(self, population, ranges):
        new_population = []
        while len(new_population) < self.pop_size:
            # Select two parents
            # parent1 = self.select_parent(population, ranges)
            # parent2 = self.select_parent(population, ranges)
            parent1 = self.select_parent_tournament(population)
            parent2 = self.select_parent_tournament(population)
            # Crossover
            if np.random.random() < CROSSOVER_PROB:
                child1, child2 = self.partially_matched_crossover(parent1, parent2)
                # child1, child2 = self.crossover(parent1, parent2)
                # child1, child2 = self.caterplilar_crossover(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)
            else:
                new_population.append(parent1)
                new_population.append(parent2)
        return self.mutate_population(new_population)

    def select_parent(self, population, ranges):
        rand = np.random.random()
        previous = 0
        for i in range(len(ranges)):
            if ranges[i] > rand >= previous:
                # print("selected " + str(i) + " index parent")
                return population[i]
            previous = ranges[i]
        return population[-1]  # Should never happen

    def crossover(self, parent1, parent2):
        assert len(parent1.chromosome) == len(parent2.chromosome)
        child1 = Candidate()
        child2 = Candidate()
        child1_chromosome = [-1] * len(parent1.chromosome)
        child2_chromosome = [-1] * len(parent2.chromosome)
        random_index = np.random.randint(0, len(parent2.chromosome) - 1)

        for i in range(len(parent1.chromosome)):
            if i < random_index:
                child1_chromosome[i] = parent1.chromosome[i]
                child2_chromosome[i] = parent2.chromosome[i]
            else:
                if parent2.chromosome[i] not in child1_chromosome:
                    child1_chromosome[i] = parent2.chromosome[i]
                else:
                    child1_chromosome[i] = self.add_first_not_existing(child1_chromosome)
                if parent1.chromosome[i] not in child2_chromosome:
                    child2_chromosome[i] = parent1.chromosome[i]
                else:
                    child2_chromosome[i] = self.add_first_not_existing(child2_chromosome)
        child1.chromosome = child1_chromosome
        child2.chromosome = child2_chromosome
        return child1, child2

    def mutate_population(self, population):
        for candidate in population:
            if np.random.random() < MUTATION_PROB:
                self.mutate(candidate)
        return population

    def mutate(self, candidate):
        index1 = np.random.randint(0, len(candidate.chromosome) - 1)
        index2 = np.random.randint(0, len(candidate.chromosome) - 1)
        temp = candidate.chromosome[index1]
        candidate.chromosome[index1] = candidate.chromosome[index2]
        candidate.chromosome[index2] = temp

    def print_population(self, population):
        for candidate in population:
            print(candidate.chromosome, candidate.fitness)

    def add_first_not_existing(self, chromosome):
        for i in range(self.number_of_products):
            if i not in chromosome:
                return i
        return -1

    def partially_matched_crossover(self, parent1, parent2):
        size = len(parent1.chromosome)
        crossover_point1, crossover_point2 = np.random.randint(0, len(parent2.chromosome) - 1), \
            np.random.randint(0, len(parent2.chromosome) - 1)

        offspring_1_chromosome = [None] * size
        offspring_2_chromosome = [None] * size
        offspring_1 = Candidate()
        offspring_2 = Candidate()

        # Copy segments
        offspring_1_chromosome[crossover_point1:crossover_point2 + 1] = parent2.chromosome[
                                                                        crossover_point1:crossover_point2 + 1]
        offspring_2_chromosome[crossover_point1:crossover_point2 + 1] = parent1.chromosome[
                                                                        crossover_point1:crossover_point2 + 1]

        def fill_offspring(offspring, parent, other_parent, mapping):
            for i in range(size):
                if offspring[i] is None:
                    value = parent[i]
                    while value in offspring:
                        value = mapping[value]
                    offspring[i] = value

        # Create mapping for duplicates
        mapping1 = {parent1.chromosome[i]: parent2.chromosome[i] for i in range(size)}
        mapping2 = {parent2.chromosome[i]: parent1.chromosome[i] for i in range(size)}

        # Fill the offspring with the unique values from the other parent
        fill_offspring(offspring_1_chromosome, parent1.chromosome, parent2.chromosome, mapping1)
        fill_offspring(offspring_2_chromosome, parent2.chromosome, parent1.chromosome, mapping2)
        offspring_1.chromosome = offspring_1_chromosome
        offspring_2.chromosome = offspring_2_chromosome

        return offspring_1, offspring_2

    def caterplilar_crossover(self, parent1, parent2):
        left = right = []
        left_index = 0
        right_index = len(parent1.chromosome) - 1
        flag_left = flag_right = False
        while left_index < right_index and (not flag_left or not flag_right):
            left.append(parent1.chromosome[left_index])
            right.append(parent1.chromosome[right_index])
            print(right_index, left_index)
            if parent2.chromosome[left_index] not in left:
                flag_left = True
            if parent2.chromosome[right_index] not in right:
                flag_right = True
            left_index += 1
            right_index -= 1

        offspring1 = Candidate()
        offspring2 = Candidate()
        print(parent1.chromosome)
        print(parent2.chromosome)
        offspring1.chromosome = parent1.chromosome[0:left_index] + parent2.chromosome[
                                                                   left_index:right_index + 1] + parent1.chromosome[
                                                                                                 right_index + 1:]
        offspring2.chromosome = parent2.chromosome[0:left_index] + parent1.chromosome[
                                                                   left_index:right_index + 1] + parent2.chromosome[
                                                                                                 right_index + 1:]
        print(offspring1.chromosome)
        print(offspring2.chromosome)
        return offspring1, offspring2

    def select_parent_tournament(self, population):
        tournament_size = 2  # Set the tournament size to 2
        tournament_contestants = np.random.randint(0, len(population) - 1, tournament_size)
        return population[min(tournament_contestants)]
