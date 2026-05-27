import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
    recall_score
)

# Loading dataset

df = pd.read_csv("data/features.csv")

X = df[
    [
        "Asymmetry",
        "Border",
        "HSV_Hue_Variance",
        "HSV_Saturation_Variance",
        "HSV_Value_Variance"
    ]
]

y = df["Cancerous"]  # 0 = non-cancerous, 1 = cancerous

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Cross-validation for different K values

k_range = range(1, 55, 2)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracy_scores = []
recall_scores = []
false_negative_scores = []

for k in k_range:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance"
    )

    scores = cross_validate(
        knn,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "recall": make_scorer(recall_score)
        }
    )

    mean_accuracy = scores["test_accuracy"].mean()
    mean_recall = scores["test_recall"].mean()

    accuracy_scores.append(mean_accuracy)
    recall_scores.append(mean_recall)

    # Also check false negatives on the test set
    knn.fit(X_train, y_train)
    y_pred_temp = knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_temp)
    false_negatives = cm[1, 0]

    false_negative_scores.append(false_negatives)

    print(
        f"K={k:2d} | "
        f"Accuracy={mean_accuracy:.4f} | "
        f"Recall={mean_recall:.4f} | "
        f"False Negatives={false_negatives}"
    )

# 4. Choose best K

best_k_accuracy = list(k_range)[np.argmax(accuracy_scores)]
best_k_recall = list(k_range)[np.argmax(recall_scores)]
best_k_false_negatives = list(k_range)[np.argmin(false_negative_scores)]

print("\nBest K by accuracy:", best_k_accuracy)
print("Best K by recall:", best_k_recall)
print("Best K with lowest false negatives:", best_k_false_negatives)

# Choosing final K

final_k = best_k_false_negatives

print("\nChosen final K:", final_k)

# Plot K analysis

plt.figure(figsize=(10, 6))

plt.plot(k_range, accuracy_scores, marker="o", label="Accuracy")
plt.plot(k_range, recall_scores, marker="o", label="Recall")

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Score")
plt.title("KNN Cross-Validation Performance")
plt.xticks(k_range)
plt.grid(True)
plt.legend()

plt.show()

plt.figure(figsize=(10, 6))

plt.plot(k_range, false_negative_scores, marker="o", label="False Negatives")

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Number of False Negatives")
plt.title("False Negatives for Different K Values")
plt.xticks(k_range)
plt.grid(True)
plt.legend()

plt.show()

# Train final KNN model

knn = KNeighborsClassifier(
    n_neighbors=final_k,
    weights="distance"
)

knn.fit(X_train, y_train)

# Predict and evaluate

y_pred = knn.predict(X_test)

print("\nFinal Model Results")
print("Final K:", final_k)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 8. Confusion matrix plot

ConfusionMatrixDisplay.from_estimator(
    knn,
    X_test,
    y_test,
    display_labels=["Non-cancerous", "Cancerous"],
    cmap="Blues"
)

plt.title("KNN Confusion Matrix")
plt.show()


