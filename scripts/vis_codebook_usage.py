import numpy as np
import matplotlib.pyplot as plt

# Load the npy file
A = np.load('tmp.npy')
# Sort A in descending order and convert to frequencies
A_sorted = np.sort(A)[::-1]
A_freq = A_sorted / np.sum(A_sorted)
# A_freq = A_sorted

# Create the x-ticks as powers of 2
N = len(A_freq)
x_ticks = list(range(N))

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.plot(x_ticks, A_freq)

# Set x-axis to display powers of 2
plt.xscale('log', base=2)
plt.xlabel('')
plt.yscale('log', base=10)
plt.ylabel('Frequency')
plt.title('Codebook Usage')

# Save the figure as res.png
plt.savefig('res.png')

plt.show()
