import pandas as pd
import pickle
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix path so we can import clean_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.text_cleaning import clean_text

# Load dataset
df = pd.read_csv('data/whatsapp_complaints_new.csv')

# Clean the complaint text
df['Cleaned_Text'] = df['Complaint_Text'].astype(str).apply(clean_text)

# Target columns and filenames
targets = {
    'category_model.pkl': 'Category',
    'delay_reason_model.pkl': 'Delay_Reason',
    'delayed_by_model.pkl': 'Delayed_By',
    'agency_model.pkl': 'Agency_Responsible',
    'resolution_model.pkl': 'Resolution_Status',
    'severity_model.pkl': 'Complaint_Severity',
}

# Loop through each target and train model
for filename, target in targets.items():
    print(f"Training model for: {target}")

    # Drop rows with missing data
    subset_df = df[['Cleaned_Text', target]].dropna()

    X = subset_df['Cleaned_Text']
    y = subset_df[target]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    # Save both model and vectorizer together
    with open(f'models/{filename}', 'wb') as f:
        pickle.dump((model, vectorizer), f)

print("✅ All models trained and saved successfully.")
