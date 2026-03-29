#Chapter 6, Example 6.2: Expected value of a multi-dimensional auto revenue
import numpy as np
import scipy.stats as stats

# Input data
a, b, = 20, 3 # For D1 (silver) # Mean and standard deviation for demand
c, d = 15, 5 # For D2 (black)
e, f = 8, 4 # For D3 (blue)
r = 25000 # Base price for vehicles
I1, I2, I3 = 50, 40, 30 # Available inventory for silver, black, blue vehicles

# Monte Carlo sampling
nsamples = 100000 # Number of samples
np.random.seed(42) #Fix seed for repeatability

# Define the revenue function
def revenue(D1, D2, D3):
    revenue_silver = r * np.minimum(D1, I1)
    revenue_black = r * np.minimum(D2, I2)
    revenue_blue = r * np.minimum(D3, I3)
    shift_silver_black = 0.9*r*np.minimum(np.maximum(I2-D2,0),np.maximum(D1-I1,0))
    shift_black_blue = 0.92*r*np.minimum(np.maximum(I3-D3,0),np.maximum(D2-I2,0))
    total_revenue = (revenue_silver + revenue_black + revenue_blue
                    + shift_silver_black + shift_black_blue)
    return total_revenue

# Generate demand samples for each color
sample = np.random.normal(loc=[a, c, e], scale=[b, d, f], size=(nsamples, 3))

# Mean revenue estimate
revenues = np.array([revenue(*demand) for demand in sample])

# Compute the standard deviation and standard error
std_dev = np.std(revenues, ddof=1)
std_error = std_dev / np.sqrt(nsamples)

# Mean revenue and 95% confidence interval
mean = np.mean(revenues)
CI = stats.norm.interval(0.95, loc=mean, scale=std_error)

# Print results 
print(f"Mean Revenue: ${mean:.2f}")
print(f"95% Confidence Interval: ${CI[0]:.2f} to ${CI[1]:.2f}")