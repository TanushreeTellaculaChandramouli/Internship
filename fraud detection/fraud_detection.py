import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = pd.read_csv('/Users/tanushree/Desktop/Internship/fraud detection/fraud_detection_sample.csv')
print("First 5 rows of dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
df_encoded = pd.get_dummies(df, drop_first=True)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='is_fraud')
plt.title('Fraud vs Non-Fraud Count')
plt.show()
X = df_encoded.drop('is_fraud', axis=1)
y = df_encoded['is_fraud']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
