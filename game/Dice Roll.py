import numpy as np
import pandas as pd
rolls = np.random.randint(1, 7, size=1000)
unique, counts = np.unique(rolls, return_counts=True)

for val, count in zip(unique, counts):
    print(f"Face {val}: {count} times")

print("Most common face:", unique[np.argmax(counts)])