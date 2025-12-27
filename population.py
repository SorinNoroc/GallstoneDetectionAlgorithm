import numpy as np
from chromosome import Chromosome
from typing import List


def tournament(population: List[Chromosome], k=3) -> Chromosome:
    selected = np.random.choice(population, size=k, replace=False)
    return get_best(selected)


def get_best(population: List[Chromosome]) -> Chromosome:
    return max(population, key=lambda x: x.fitness).copy()


def crossover(parent1: Chromosome, parent2: Chromosome, rate=0.5) -> Chromosome:
    child = parent1.copy()
    child.genes = rate * parent1.genes + (1.0 - rate) * parent2.genes
    return child


def reset_mutation(c: Chromosome, rate=0.1):
    for i in range(c.n_genes):
        if np.random.random() < rate:
            c.genes[i] = c.min_vals[i] + np.random.random() * (c.max_vals[i] - c.min_vals[i])
