import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

st.title("❤️ Heart Disease Prediction Dashboard")

# Load dataset
df = pd.read_csv("heart.csv")

st.subheader("📊 Dataset Overview")
st.write(df.head())

# Correlation
st.subheader("🔍 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Data preparation
X = df.drop('output', axis=1)
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

# Model performance
st.subheader("🏁 Model Accuracy Comparison")
st.write(pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']))

fig2, ax2 = plt.subplots()
sns.barplot(x=list(results.keys()), y=list(results.values()), ax=ax2)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
st.pyplot(fig2)

# selection
st.subheader("🩺 Make a Prediction")
selected_model_name = st.selectbox("Choose a model:", list(models.keys()))
model = models[selected_model_name]

# User inputs
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trtbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalachh = st.number_input("Maximum Heart Rate Achieved", 70, 210, 150)
exng = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
slp = st.selectbox("Slope of the Peak Exercise ST Segment (0–2)", [0, 1, 2])
caa = st.number_input("Number of Major Vessels (0–4)", 0, 4, 0)
thall = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# prediction
if st.button("🔍 Predict"):
    user_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg,
                           thalachh, exng, oldpeak, slp, caa, thall]])
    user_data = scaler.transform(user_data)
    prediction = model.predict(user_data)

    if prediction[0] == 1:
        st.error(f"⚠️ {selected_model_name} predicts a **high risk** of heart disease.")
    else:
        st.success(f"✅ {selected_model_name} predicts a **low risk** of heart disease.")

# evaluation section
st.subheader("📈 Model Evaluation (Confusion Matrix & ROC Curve)")
if st.button("Show Evaluation"):
    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title(f"Confusion Matrix - {selected_model_name}")
    st.pyplot(fig3)

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig4, ax4 = plt.subplots()
        ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax4.set_xlim([0.0, 1.0])
        ax4.set_ylim([0.0, 1.05])
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title(f'ROC Curve - {selected_model_name}')
        ax4.legend(loc="lower right")
        st.pyplot(fig4)
    else:
        st.warning(f"The selected model ({selected_model_name}) does not support probability outputs.")

