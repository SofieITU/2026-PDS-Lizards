import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


df1 = pd.read_csv('../../data/features.csv', index_col=0)
df2 = pd.read_csv('../../data/features_SMOTE.csv', index_col=0)

# Stack vertically
combined = pd.concat([df1, df2], axis=0, ignore_index=False)

# Renumber indices if needed
combined = combined.reset_index(drop=True)

combined.to_csv('../../data/SMOTE.csv')


combined_df = pd.read_csv("../../data/SMOTE.csv")
combined_df.columns = combined_df.columns.astype(str)


X = combined_df.drop(columns=['Cancerous', 'ID', 'Unnamed: 0'], errors='ignore')
y = combined_df['Cancerous']
image_ids = combined_df["ID"]

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, image_ids, test_size=0.2, stratify=y, random_state=42)


n_est_metrics = {"train":[], "val":[]}
max_depth_metrics = {"train":[], "val":[]}
n_estimator = [1, 10, 25, 50, 100, 250]
depth_list = [3, 4, 5, 6, 7, 8, None]

for n in n_estimator:
    rf = RandomForestClassifier(n_estimators=n, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    train_acc = rf.score(X_train, y_train)
    val_acc = rf.score(X_test, y_test)
    n_est_metrics["train"].append(train_acc)
    n_est_metrics["val"].append(val_acc)
    print(f"n_estimators = {n} - train_acc: {train_acc:.3f} - val_acc: {val_acc:.3f}")

print("")
for d in depth_list:
    rf = RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=42, max_depth=d)
    rf.fit(X_train, y_train)
    train_acc = rf.score(X_train, y_train)
    val_acc = rf.score(X_test, y_test)
    max_depth_metrics["train"].append(train_acc)
    max_depth_metrics["val"].append(val_acc)
    print(f"max_depth_metrics = {d} - train_acc: {train_acc:.3f} - val_acc: {val_acc:.3f}")


fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(n_estimator, n_est_metrics["train"], marker='o', label="Training Score", color="blue")
ax[0].plot(n_estimator, n_est_metrics["val"], marker='s', label="Validation Score", color="orange")
ax[0].set_title("Validation Curve: Number of Trees")
ax[0].set_xlabel("Number of estimators")
ax[0].set_ylabel("Accuracy")
ax[0].legend()
ax[0].grid(True, linestyle='--', alpha=0.6)

depth_labels = [str(d) for d in depth_list]

ax[1].plot(depth_labels, max_depth_metrics["train"], marker='o', label="Training Score", color="blue")
ax[1].plot(depth_labels, max_depth_metrics["val"], marker='s', label="Validation Score", color="orange")
ax[1].set_title("Validation Curve: Max Depth")
ax[1].set_xlabel("max_depth")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
ax[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


final_estimators = [10, 25, 50]
final_depth = [3, 4, 5, 6, 7]

for n in final_estimators:
    for d in final_depth:
        rf_final = RandomForestClassifier(n_estimators=n, class_weight='balanced', random_state=42, max_depth=d)
        rf_final.fit(X_train, y_train)

        final_preds = rf_final.predict(X_test)

        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, final_preds), annot=True, fmt='d', cmap='Blues')
        plt.title(f"Random Forest Classifier SMOTE\nConfusion Matrix\n(0=Benign, 1=Cancer)\n Nr. of estimator: {n}, depth: {d}")
        plt.ylabel('Actual Truth')
        plt.xlabel(f'Model Prediction\nScore: {rf_final.score(X_test, y_test)}')
        plt.savefig(f"../predictions/random_forest_classifier_SMOTE/conf_matrix_{n}_estim_{d}_depth.png", dpi = 200, bbox_inches = "tight")
        

def generate_submission_csv(trained_model, test_features, test_ids, output_filename):
    """
    Takes any trained model, generates predictions and probabilities, 
    and saves them in csv format.
    """
    labels = trained_model.predict(test_features)
    probabilities = trained_model.predict_proba(test_features)[:, 1]
    
    submission_df = pd.DataFrame({
        'image_id': test_ids,
        'label': labels,
        'probability': probabilities
    })
    
    submission_df['patient_id'] = submission_df['image_id'].apply(
        lambda x: "_".join(str(x).split("_")[:2])
    )
    
    submission_df = submission_df[['image_id', 'patient_id', 'label', 'probability']]
    
    submission_df.to_csv(output_filename, index=False)
    
    return submission_df


predictions_csv = generate_submission_csv(rf_final, X_test, id_test, "../predictions/random_forest_classifier_SMOTE/RF_predictions.csv")

predictions_csv.head()