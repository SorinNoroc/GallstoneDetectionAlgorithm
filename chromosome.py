import numpy as np

class Chromosome:
    def __init__(self, n_genes, min_vals, max_vals):
        self.n_genes = n_genes
        self.min_vals = np.array(min_vals, dtype=float)
        self.max_vals = np.array(max_vals, dtype=float)

        # Initialise genes to random values between min_vals and max_vals
        self.genes = self.min_vals + np.random.rand(n_genes) * (self.max_vals - self.min_vals)

        self.fitness = None


    # Function to make a deep copy of the best chromosome
    def copy(self):
        clone = Chromosome(self.n_genes, self.min_vals, self.max_vals)
        clone.genes = self.genes.copy()
        clone.fitness = self.fitness
        return clone