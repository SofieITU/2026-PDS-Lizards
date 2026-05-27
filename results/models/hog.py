import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def generate_submission_csv(trained_model, test_features, test_ids, output_filename):
    """
    Takes any trained model, generates predictions and probabilities, 
    and saves them in the exact required CSV format.
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

combined_df = pd.read_csv("../../data/hog_features.csv")
combined_df.columns = combined_df.columns.astype(str)


X = combined_df.drop(columns=['Cancerous', 'ID', 'Unnamed: 0'], errors='ignore')
y = combined_df['Cancerous']
image_ids = combined_df["ID"]

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, image_ids, test_size=0.2, stratify=y, random_state=42)

rf_final = RandomForestClassifier(n_estimators=250, class_weight='balanced', random_state=42, max_depth=6)
rf_final.fit(X_train, y_train)

final_preds = rf_final.predict(X_test)

predictions_csv = generate_submission_csv(rf_final, X_test, id_test, "../predictions/hog/RF_with_HOG_predictions.csv")