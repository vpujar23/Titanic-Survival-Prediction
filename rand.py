import matplotlib.pyplot as plt
import numpy as np

# Generate some data
np.random.seed(0)
data = np.random.rand(5)

# Sort the data
# sorted_data = sorted(data)

# Create a bar plot with sorted data
plt.bar(range(len(sorted_data)), sorted_data)
plt.show()
