import matplotlib.pyplot as plt
from scipy.stats import binom

# Model names and probabilities
models = {
    "Q3B_mi": 0.7009,
    "Q7B_mi": 0.7297,
    "Koshkina": 0.8338,
}

# Numbers of extractions
n_values = [3, 5, 9, 13, 21,35]

# Compute probability of majority of A for each model
results = {}
for name, p in models.items():
    probs = []
    for n in n_values:
        majority = n // 2 + 1
        prob_majority = 1 - binom.cdf(majority - 1, n, p)
        probs.append(prob_majority)
    results[name] = probs

# Plot results
plt.figure(figsize=(8, 5))
for name, probs in results.items():
    plt.plot(n_values, probs, marker='o', linewidth=2, label=f"{name} (p={models[name]:.4f})")

plt.title("Probability of Majority of Correct Classifications vs Number of Extractions", fontsize=13)
plt.xlabel("Number of Extractions", fontsize=11)
plt.ylabel("P(Majority of Correct Classifications)", fontsize=11)
# plt.ylim(0.925, 1.02)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
