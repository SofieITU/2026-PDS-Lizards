import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
def main(features_path, prediction_results_path, model_path, load_model):
    """
    Docstring for main
    
    :param features_path: Path to the features csv used as input to the model (e.g. ./data/features.csv).
    :param prediction_results_path: Path to save the output predictions of the model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param model_path: Path to save or load the trained model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param load_model: Boolean to train the model and save it to model_path if False, load it from model_path if True. 
    """
    
    # load dataset CSV file
    df = pd.read_csv(features_path)
    df.columns = df.columns.astype(str)
    X = df.drop(columns=['Cancerous', 'ID', 'Unnamed: 0'], errors='ignore')
    y = df['Cancerous']
    image_ids = df["ID"]

    # split the dataset into training and testing sets.
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, image_ids, test_size=0.2, stratify=y, random_state=42)
    if load_model:
        # load the model
        model = joblib.load(model_path)
    else:
        # train the classifier (using logistic regression as an example)
        model = RandomForestClassifier(n_estimators=250, max_depth=6, class_weight="balanced",random_state=42)
        model.fit(X_train, y_train)

        # save the model.
        joblib.dump(model, model_path)
        print("Training completed!")

    # test the classifier.
    model_predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:,1]

    # write test results to CSV.
    prediction_csv = pd.DataFrame({
        "image_id" : id_test,
        "label" : model_predictions,
        "probability" : probabilities
    })
    prediction_csv["patient_id"] = prediction_csv["image_id"].apply(lambda x: "_".join(str(x).split("_")[:2]))
    prediction_csv = prediction_csv[["image_id","patient_id","label","probability"]]
    prediction_csv.to_csv(prediction_results_path, index=False)


if __name__ == "__main__":
    features_path = "data/features.csv"
    prediction_results_path = "results/predictions/predictions_MODEL.csv"
    model_path = "results/predictions/predictions_MODEL.csv"
    load_model = False

    main(features_path, prediction_results_path,model_path,load_model)