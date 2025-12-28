from chromosome import *
from IProblem import IProblem


class SVMDualProblem(IProblem):
    def __init__(self, X_train, y_train, gamma=0.014, C=1.0, kernel=None):
        self.X_train = X_train
        self.y_train = y_train
        self.n = X_train.shape[0]
        self.C = C
        self.gamma = gamma

        # Initializing the kernel function
        if kernel is None:
            self.kernel = self.rbf_kernel
        else:
            self.kernel = kernel

        # Initializing the kernel matrix
        self.kernel_matrix = self.rbf_kernel(X_train, X_train)

        # Alpha values should be between [0, C]
        self.min_vals = np.zeros(self.n, dtype=np.float64)
        self.max_vals = np.ones(self.n, dtype=np.float64) * self.C


    def make_chromosome(self):
        return Chromosome(self.n, self.min_vals, self.max_vals)


    def compute_fitness(self, chromosome: Chromosome) -> float:
        alpha = chromosome.genes
        v = alpha * self.y_train

        # Objective function for the current problem
        objective = np.sum(alpha) - 0.5 * (v @ self.kernel_matrix @ v)

        # Force to float
        chromosome.fitness = float(objective)

        return chromosome.fitness


    def repair(self, chromo, tolerance = 1e-7, max_iter = 10000):
        """
        Enforce constraints after genetic operations:
            1. 0 <= alpha <= C
            2. sum(alpha * y) == 0
        """

        # Enforce equality constraint: sum(alpha * y) == 0
        alpha, y = chromo.genes, self.y_train
        s = np.dot(alpha, y)
        it = 0

        while abs(s) > tolerance and it < max_iter:
            it += 1

            # We must adjust alphas so that sum(alpha * y) == 0
            # pos_idx: indices of positive y values
            # neg_idx: indices of negative y values

            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == -1)[0]

            # The sum of alphas where y is positive/negative
            s_plus = np.sum(alpha[pos_idx])
            s_minus = np.sum(alpha[neg_idx])

            # We choose an alpha value to trim, based on what sum is larger
            if s_plus > s_minus:
                k = np.random.choice(pos_idx)
            else:
                k = np.random.choice(neg_idx)

            """
                From the current alpha value we chose
                to either substract the entire amount
                of drift, or make it zero (it can't be
                negative)
            """
            if alpha[k] > abs(s):
                alpha[k] -= abs(s)
            else:
                alpha[k] = 0.0

            """
                Compute the constraint again, repeat the
                process until the drift is small enough,
                or the maximum nr of iterations is reached
            """
            s = np.dot(alpha, y)

        # Safety re-clipping before we exit the function
        np.clip(alpha, self.min_vals, self.max_vals, out=alpha)


    """
        Shorter, vectorised form of computing the 
        kernel matrix between 2 set of points
    """
    def rbf_kernel(self, x1, x2):
        diff = x1[:, None, :] - x2[None, :, :]
        return np.exp(-self.gamma * np.sum(diff ** 2, axis=2))
