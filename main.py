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

# ----------------- Decision values -----------------

K_sv_test = problem.rbf_kernel(X_sv, X_test)
decision_values = np.dot(alpha_sv * y_sv , K_sv_test)

# -------------------- Bias term --------------------

K_sv_sv = problem.rbf_kernel(X_sv, X_sv)
b = np.mean(y_sv - np.dot(alpha_sv * y_sv , K_sv_sv))

# ------------------- Predictions -------------------

# Function
f_x = decision_values + b

# Predictions
y_hat = np.sign(f_x)

# ---------------- Confusion Matrix -----------------
TP = np.sum( (y_hat == 1) & (y_test == 1) )
FP = np.sum( (y_hat == 1) & (y_test == -1) )
TN = np.sum( (y_hat == -1) & (y_test == -1) )
FN = np.sum( (y_hat == -1) & (y_test == 1) )

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# -------------- Write results to file --------------

result_file = "results.txt"

with open(f"outputs/{result_file}", "w") as f:
    f.write(f"Support Vector Number: {len(sv_indices)}\n\n")
    f.write(f"Support Vector Indices: {sv_indices}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(f"TP: {TP}\nFP: {FP}\nTN: {TN}\nFN: {FN}\n\n")

    f.write("Classification Report:\n")
    f.write(f"Accuracy: {accuracy*100}%\n"
            f"Precision: {precision*100}%\n"
            f"Recall: {recall*100}%\n"
            f"F1: {f1*100}%\n\n"
            )

    f.write("Decision values for all samples (besides y_test):\n")
    outputs = {"TP": [], "TN": [], "FP": [], "FN": []}

    for dv, y in zip(f_x, y_test):
        outputs[("TP" if y == 1 else "FP") if dv >= 0 else ("TN" if y == -1 else "FN")].append(dv)

    for k, v in outputs.items():
        f.write(f"{k}:\n" + "".join(f"{x:.6f}\n" for x in v) + "\n")

print(f"Results saved to outputs/{result_file}")