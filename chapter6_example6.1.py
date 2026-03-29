#Chapter 6, Example 6.1:  Coin-flip simulation
import numpy as np
import scipy.stats as stats

# Input data
nsamples = 100 # User-defined variable N = # of coin flips
np.random.seed(42) #Fix seed for repeatability

# Simulate coin flips and estimate mean value and 95% Confidence
sample = np.random.choice([1, -1], size=nsamples)
mean = np.mean(sample) # Compute mean of outcomes

std_dev = np.std(sample, ddof=1) # Compute the standard deviation
std_error = std_dev / np.sqrt(nsamples) 

CI = stats.norm.interval(0.95,loc=mean,scale=std_error) # 95% confidence interval

# Print results
print(f"N: {nsamples}")
print(f"Mean Estimate: {mean}")
print(f"95% Confidence Interval: {CI}")