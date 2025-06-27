import numpy as np
import pandas as pd
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Math": [85, 78, 92, 60],
    "Science": [89, 76, 95, 70],
    "English": [91, 80, 85, 72]
}

df = pd.DataFrame(data)
print(df)

# Average score per student
df["Average"] = df[["Math", "Science", "English"]].mean(axis=1)
print(df[["Name", "Average"]])

# Filter
print(df[df["Math"] > 85])