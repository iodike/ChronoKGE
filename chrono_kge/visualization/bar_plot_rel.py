# libraries
import numpy as np
from numpy import linspace
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# dataframe
df = pd.DataFrame({
'var1': np.random.normal(size=1000),
'var2': np.random.normal(loc=2, size=1000) * -1
})

# Fig size
plt.rcParams["figure.figsize"] = 12, 8

# plot histogram chart for var1
sns.histplot(x=df.var1, stat="density", bins=5, edgecolor='black')

# plot histogram chart for var2
n_bins = 5

# get positions and heights of bars
heights, bins = np.histogram(df.var2, density=True, bins=n_bins)

# multiply by -1 to reverse it
heights *= -1
bin_width = 0.1 #np.diff(bins)[0]
bin_pos = (bins[:-1] + bin_width / 2) * -1

# plot
plt.bar(bin_pos, heights, width=bin_width, edgecolor='black')

# show the graph
plt.show()
