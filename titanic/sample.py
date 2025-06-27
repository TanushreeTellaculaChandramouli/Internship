import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
df = pd.read_csv('titanic/test.csv')

# 1. Bar Plot - Gender Count
sns.countplot(x='Sex', data=df)
plt.title("Passenger Gender Distribution")
plt.show()

# 2. Histogram - Age Distribution
sns.histplot(df['Age'].dropna(), kde=True, bins=30)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.show()

# 3. Box Plot - Age vs Class
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Age Distribution per Passenger Class")
plt.xlabel("Passenger Class")
plt.show()

# 4. Scatter Plot - Fare vs Age
sns.scatterplot(x='Age', y='Fare', data=df)
plt.title("Fare Paid vs Passenger Age")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()


# 6. Multi-plot Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) Bar plot - Class Count
sns.countplot(x='Pclass', data=df, ax=axes[0, 0])
axes[0, 0].set_title("Passenger Class Count")

# (2) Box plot - Age by Gender
sns.boxplot(x='Sex', y='Age', data=df, ax=axes[0, 1])
axes[0, 1].set_title("Age Distribution by Gender")

# (3) Histogram - Fare
sns.histplot(df['Fare'].dropna(), kde=True, bins=30, ax=axes[1, 0])
axes[1, 0].set_title("Fare Distribution")

# (4) Scatter plot - Age vs Fare (no survival info)
sns.scatterplot(x='Age', y='Fare', data=df, ax=axes[1, 1])
axes[1, 1].set_title("Age vs Fare")

plt.tight_layout()
plt.show()
