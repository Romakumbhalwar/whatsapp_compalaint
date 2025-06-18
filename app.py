import streamlit as st
import pandas as pd
import pickle
from preprocessing.text_cleaning import clean_text
from visualization.visual_utils import generate_wordcloud, plot_bar

st.set_page_config(page_title="WhatsApp Complaint Analyzer", layout="wide")
st.title("📱 WhatsApp Complaint Analyzer")

# Load and clean data
df = pd.read_csv('data/whatsapp_complaints_new.csv')
df['Cleaned_Text'] = df['Complaint_Text'].apply(clean_text)

# Fix Arrow serialization issue for Streamlit compatibility
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)

# Sidebar menu
task = st.sidebar.selectbox("Choose Task", ["Overview", "Text Classification", "Visualization"])

# Load all models with caching
@st.cache_resource
def load_models():
    models = {}
    paths = {
        'Category': 'models/category_model.pkl',
        'Delay_Reason': 'models/delay_reason_model.pkl',
        'Delayed_By': 'models/delayed_by_model.pkl',
        'Agency_Responsible': 'models/agency_model.pkl',
        'Resolution_Status': 'models/resolution_model.pkl',
        'Complaint_Severity': 'models/severity_model.pkl'
    }
    for label, path in paths.items():
        with open(path, 'rb') as f:
            model, vectorizer = pickle.load(f)
            models[label] = (model, vectorizer)
    return models

models = load_models()

# Overview Section
if task == "Overview":
    st.subheader("📋 Full Dataset")
    st.dataframe(df)

    st.markdown("### 🔢 Column Summary:")
    st.write(df.describe(include='all'))

    st.markdown("### 📌 Missing Values:")
    st.write(df.isnull().sum())

# Prediction Section
elif task == "Text Classification":
    st.subheader("🔍 Predict Complaint Fields")
    user_input = st.text_area("Enter Complaint Text")
    if st.button("Predict"):
        cleaned = clean_text(user_input)
        for label, (model, vectorizer) in models.items():
            X = vectorizer.transform([cleaned])
            pred = model.predict(X)[0]
            st.write(f"**{label}**: {pred}")

# Visualization Section
elif task == "Visualization":
    st.subheader("📊 Complaint Trends")
    col = st.selectbox("Select Column", df.columns[6:])
    plot = plot_bar(df, col)
    st.pyplot(plot)

    st.subheader("☁️ WordCloud of Complaints")
    wc_img = generate_wordcloud(df['Cleaned_Text'])
    st.image(wc_img)
