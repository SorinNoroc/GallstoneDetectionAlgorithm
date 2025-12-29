from population import *
from SVMDualProblem import *


class EvolutionaryAlgorithm:
    def __init__(self, problem: IProblem, population_size=200, max_generations=2000, crossover_rate=0.5, mutation_rate=0.1, threshold=0.5):
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.threshold = threshold

    def solve(self):

        population = [self.problem.make_chromosome() for _ in range(self.population_size)]

        for chromosome in population:
            self.problem.repair(chromosome)
            self.problem.compute_fitness(chromosome)

        # Used for earlier stopping if threshhold is smaller
        checkpoint_value = None

        for generation in range(self.max_generations):
            new_population = [get_best(population)]

            for i in range(1, self.population_size):
                mother = tournament(population)
                father = tournament(population)
                while father is mother:
                    father = tournament(population)

                child = crossover(mother, father, self.crossover_rate)
                reset_mutation(child, self.mutation_rate)

                self.problem.repair(child)
                self.problem.compute_fitness(child)

                new_population.append(child)

            population = new_population
            best = new_population[0]

            # Check the threshold
            if generation % 500 == 0:
                if checkpoint_value is None:
                    checkpoint_value = best.fitness
                else:
                    if best.fitness < checkpoint_value + self.threshold:
                        return get_best(population)
                    checkpoint_value = best.fitness

            # Print training results each 10 runs
            if generation % 10 == 0:
                constraint = abs( (best.genes * self.problem.y_train).sum() )
                support_vectors_count = np.sum(best.genes > 1e-6)
                print(f"Gen: {generation:4d} | "
                      f"Best Fitness: {best.fitness:.6f} | "
                      f"Constraint: {constraint:4e} | "
                      f"Support Vectors: {support_vectors_count} | "
                      )

        return get_best(population)