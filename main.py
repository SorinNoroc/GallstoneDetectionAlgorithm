from SVMDualProblem import SVMDualProblem
from evolution import EvolutionaryAlgorithm
from dataset_functions import *
import numpy as np
from os import makedirs


# ------------- Load and normalise data --------------

X, y = load_data()

X_train, y_train, X_test, y_test = split_train_test(X, y)

mean, std = normaliser_fit(X_train)
X_train = normaliser_apply(X_train, mean, std)
X_test = normaliser_apply(X_test, mean, std)


# -------------- Initialise problem ------------------

problem = SVMDualProblem(X_train, y_train)


# -------- Initialise Evolutionary Algorithm ---------

ea = EvolutionaryAlgorithm(
    problem,
    population_size=400,
    max_generations=2000,
    crossover_rate=0.98,
    mutation_rate=0.01
)

best = ea.solve()

# --------- Extract alpha values and save -----------

alpha = best.genes
makedirs("outputs", exist_ok=True)
np.savetxt("outputs/model.txt", alpha)

# --------------- Find support vectors --------------

sv_indices = np.where(alpha > 1e-6)[0]
alpha_sv = alpha[sv_indices]
X_sv = X_train[sv_indices]
y_sv = y_train[sv_indices]