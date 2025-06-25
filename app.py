import streamlit as st
st.set_page_config(page_title="WhatsApp Complaint Analyzer", layout="centered")

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Data ----------
@st.cache_data
def load_data():
    return pd.read_csv("data/w_complaints.csv")

df = load_data()

# ---------- Rules ----------
reason_map = {
    "traffic|jam|road block": ("Late Delivery", "Traffic congestion"),
    "holiday|festival|closed|festival": ("Late Delivery", "Public holiday"),
    "courier|shipment|delivery partner": ("Late Delivery", "Courier delay"),
    "broken|damaged|cracked|damage|crak|box|bad packaging": ("Damaged Product", "Poor Packaging"),
    "rude|disrespectful|impolite|misbehaved|speaking rudely|very impolite": ("Rude Staff", "Unprofessional Staff Behavior")
}

action_map = {
    "Traffic congestion": "Escalate to delivery team",
    "Public holiday": "Update ETA & notify customer",
    "Courier delay": "Follow up with courier partner",
    "Poor Packaging": "Initiate refund/replacement & notify packaging team",
    "Unprofessional Staff Behavior": "Escalate to HR for internal review"
}

# ---------- Sidebar ----------
if st.sidebar.checkbox("Show Category Distribution"):
    st.subheader("📊 Complaint Category Distribution")
    if "Predicted_Category" in df.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x="Predicted_Category")
        plt.xticks(rotation=15)
        plt.title("Category Frequency")
        st.pyplot(plt.gcf())
    else:
        st.warning("Predicted_Category column not found in data.")

# ---------- Main UI ----------
st.title("📱 WhatsApp Complaint Analyzer")
msg = st.text_area("Enter complaint text here:")

if st.button("Analyze Complaint"):
    cat, rsn, act = "Unknown", "Unknown", "No action available"
    for pat, (c, r) in reason_map.items():
        if re.search(pat, msg.lower()):
            cat, rsn = c, r
            act = action_map.get(rsn, act)
            break
    st.markdown("### 🧾 Prediction Result")
    st.write(f"**Complaint Text**: {msg}")
    st.write(f"**Predicted Category**: {cat}")
    st.write(f"**Detected Reason**: {rsn}")
    st.write(f"**Suggested Action**: {act}")
