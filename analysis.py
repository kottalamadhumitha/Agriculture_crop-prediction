import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

from features import ALL_FEATURE_NAMES

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Crop_recommendation.csv")
df.columns = df.columns.str.strip().str.lower()

plt.style.use("default")

# -----------------------------
# 1️⃣ Crop Demand Distribution
# -----------------------------
plt.figure(figsize=(10,5))
df['label'].value_counts().plot(kind='bar', color='green')
plt.title("Crop Demand Distribution")
plt.xlabel("Crop")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 2️⃣ Temperature Trend by Crop
# -----------------------------
plt.figure(figsize=(12,5))
sns.boxplot(x='label', y='temperature', data=df)
plt.xticks(rotation=90)
plt.title("Temperature Distribution Across Crops")
plt.tight_layout()
plt.show()

# -----------------------------
# 3️⃣ Rainfall Trend by Crop
# -----------------------------
plt.figure(figsize=(12,5))
sns.boxplot(x='label', y='rainfall', data=df)
plt.xticks(rotation=90)
plt.title("Rainfall Distribution Across Crops")
plt.tight_layout()
plt.show()

# -----------------------------
# 4️⃣ Season Distribution
# -----------------------------
def get_season(temp):
    if temp < 20:
        return "Winter"
    elif temp < 30:
        return "Monsoon"
    else:
        return "Summer"

df["season"] = df["temperature"].apply(get_season)

plt.figure(figsize=(6,4))
df["season"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["#95d5b2","#74c69d","#40916c"])
plt.title("Season Distribution")
plt.ylabel("")
plt.show()

# -----------------------------
# 5️⃣ Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Greens")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# 6️⃣ Feature Importance (From Model)
# -----------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))

    importances = model.feature_importances_
    names = list(ALL_FEATURE_NAMES)
    if len(names) != len(importances):
        names = [f"f{i}" for i in range(len(importances))]

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,5))
    plt.title("Feature Importance (Model Based)")
    plt.bar(range(len(importances)),
            importances[indices],
            color="green")
    plt.xticks(range(len(importances)),
               [names[i] for i in indices],
               rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

except Exception:
    print("Model not found. Train model first to see feature importance.")
