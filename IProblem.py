from chromosome import Chromosome

class IProblem:
    def make_chromosome(self) -> Chromosome:
        pass

    def compute_fitness(self, chromosome: Chromosome) -> float:
        pass

    def repair(self, chromosome: Chromosome):
        pass