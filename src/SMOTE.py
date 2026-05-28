import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

df = pd.read_csv('../data/features.csv', index_col=0)  # Insert own path to features.csv

print("Original dataset:")
print(df['Cancerous'].value_counts())
print(f"Total samples: {len(df)}\n")

# Drop ID columnm
X = df.drop(['ID', 'Cancerous'], axis=1) 
y = df['Cancerous']

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print("\nAfter SMOTE:")
print(f"Total samples: {len(X_smote)}")
print(y_smote.value_counts())

# Create NEW SMOTE dataframe 
SMOTE_df = pd.DataFrame(X_smote, columns=X.columns)
SMOTE_df['Cancerous'] = y_smote

# Add ID column back
SMOTE_df.insert(0, 'ID', range(1, len(SMOTE_df) + 1))

# Save to CSV
SMOTE_df.to_csv('../data/features_SMOTE.csv', index=False)