#Chapter 8, Example 8.4: Monte carlo policy iteration for grid world
import numpy as np
import matplotlib.pyplot as plt

# Input data
nsamples = 300
initial_mean = 5.0
mean_drift_per_step = 0.01  # Drift added to the mean at each step
std_dev = 1.0               # Standard deviation remains fixed
alpha_values = [0.01, 0.1, 0.5]  # Constant stepsizes for nonstationary settings
np.random.seed(42)               # Fix seed for repeatability

# Generate nonstationary samples
np.random.seed(42) #Fix seed for repeatability
mean = initial_mean + mean_drift_per_step * np.arange(nsamples)
sample = np.random.normal(loc=mean, scale=std_dev)

Y = [[0.0] for _ in alpha_values]# One trajectory per mu; start from 0

# Robbins–Monro updates
for i, alpha in enumerate(alpha_values):
    for m in range(nsamples):
        x_m = sample[m]
        y_prev = Y[i][-1]
        y_new = (1 - alpha) * y_prev + alpha * x_m
        Y[i].append(y_new)

# Plot results 
colors = ["tab:red", "tab:blue", "tab:gray"]
plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alpha_values):
    plt.plot(Y[i][1:],linewidth=1,color=colors[i],label=fr'$Y^m$ (alpha={alpha})')
plt.plot(mean, label='True mean (nonstationary)', linestyle='--', linewidth=2)
plt.xlabel('Iteration $m$')
plt.ylabel('Mean estimate')
plt.title('Robbins–Monro Tracking of a Drifting Mean')
plt.legend()
plt.show()
