# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

import pickle

pd.set_option('display.max_columns', 12)
print("Libraries imported successfully")

# Cell 2: Load Data
# Load datasets (use correct path)
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print("Train shape:", train.shape)   # (891, 12)
print("Test shape :", test.shape)    # (418, 11)

# Keep PassengerId for submission
passenger_ids = test['PassengerId'].copy()

# Cell 3: Basic EDA + Visualizations
print("Survival Rate:", round(train['Survived'].mean()*100, 2), "%")

# Visualizations
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('Survival by Gender')

plt.subplot(1,2,2)
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('Survival by Passenger Class')

plt.tight_layout()
plt.show()

# Cell 4: Preprocessing
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Combine train & test for consistent preprocessing
combined = pd.concat([train[features + ['Survived']], test[features]], axis=0, ignore_index=True)

# Fill missing values
combined['Age']      = combined['Age'].fillna(combined['Age'].median())
combined['Fare']     = combined['Fare'].fillna(combined['Fare'].median())
combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])

# Encode Sex
le_sex = LabelEncoder()
combined['Sex'] = le_sex.fit_transform(combined['Sex'])

# One-hot encoding for Embarked
combined = pd.get_dummies(combined, columns=['Embarked'], drop_first=True)

# Split back
train_clean = combined[combined['Survived'].notna()].copy()
test_clean  = combined[combined['Survived'].isna()].drop('Survived', axis=1)

X = train_clean.drop('Survived', axis=1)
y = train_clean['Survived'].astype(int)

print("Features used:", X.columns.tolist())

# Cell 5: Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}")

# Cell 6: Train Model
model = LogisticRegression(
    max_iter=300,
    random_state=42,
    solver='lbfgs'
)

model.fit(X_train, y_train)
print("Model trained successfully!")

# Cell 7: Evaluate Model
y_pred_val = model.predict(X_val)

acc = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_val, y_pred_val, target_names=['Died', 'Survived']))

# Cell 8: Save Model (model.pkl)
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as model.pkl")
 
import os
print("your file was saved here:", os.getcwd())