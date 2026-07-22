import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Customer Intelligence OS", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CORPORATE CSS THEME ---
st.markdown("""
<style>
    h1, h2, h3 {
        color: #1E3A8A !important; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #152C69;
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
    }
    /* Custom container styling for alerts to match the theme */
    .risk-alert {
        padding: 1rem; border-radius: 0.5rem; background-color: #FEE2E2; color: #991B1B; border: 1px solid #F87171;
    }
    .safe-alert {
        padding: 1rem; border-radius: 0.5rem; background-color: #DCFCE7; color: #166534; border: 1px solid #4ADE80;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER & BRANDING ---
st.title("📊 Customer Behavior Analytics & Predictive Engagement System")
st.markdown("**Machine Learning Semester Project**")

st.markdown("---")

# --- LOAD MODELS ---
@st.cache_resource
def load_artifacts():
    dt_model = joblib.load('models/decision_tree_model.pkl')
    kmeans_model = joblib.load('models/kmeans_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return dt_model, kmeans_model, scaler

dt_model, kmeans_model, scaler = load_artifacts()

# --- SIDEBAR INPUTS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.header("Configure Customer Profile")

with st.sidebar.container(border=True):
    tenure = st.slider("Tenure (Months)", 1, 72, 24)
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 75.0)
    total_charges = tenure * monthly_charges

with st.sidebar.container(border=True):
    st.subheader("Demographics & Account")
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    partner = st.selectbox("Has Partner?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    dependents = st.selectbox("Has Dependents?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    contract_encoded = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]

# --- MODEL EXPLANATION EXPANDER ---
with st.expander("⚙️ How It Works & Model Performance"):
    st.markdown("""
    **Methodology:**
    * **Decision Tree (Classification):** Predicts churn risk by analyzing historical patterns in customer demographics, tenure, and financial data.
    * **K-Means (Clustering):** Dynamically segments customers into distinct behavioral cohorts based on spending and loyalty metrics.
    """)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Model Accuracy", "79.0%")
    col_m2.metric("Precision", "0.61")
    col_m3.metric("Recall", "0.64")

# --- TOP KPI METRICS ---
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Customer Tenure", value=f"{tenure} Months")
    col2.metric(label="Current Monthly Rate", value=f"${monthly_charges:,.2f}")
    col3.metric(label="Lifetime Value (LTV)", value=f"${total_charges:,.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# --- MAIN DASHBOARD TABS ---
tab1, tab2 = st.tabs(["🔮 Churn Prediction", "📊 Customer Segments"])

EXPECTED_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
    'MonthlyCharges', 'TotalCharges'
]

with tab1:
    with st.container(border=True):
        st.header("Real-Time Churn Risk Analysis")
        
        scaled_nums = scaler.transform(pd.DataFrame([[tenure, monthly_charges, total_charges]], 
                                                    columns=['tenure', 'MonthlyCharges', 'TotalCharges']))
        
        input_df = pd.DataFrame(np.zeros((1, len(EXPECTED_COLUMNS))), columns=EXPECTED_COLUMNS)
        input_df['SeniorCitizen'] = senior_citizen
        input_df['Partner'] = partner
        input_df['Dependents'] = dependents
        input_df['Contract'] = contract_encoded
        input_df['tenure'] = scaled_nums[0][0]
        input_df['MonthlyCharges'] = scaled_nums[0][1]
        input_df['TotalCharges'] = scaled_nums[0][2]
        
        prediction = dt_model.predict(input_df)[0]
        probabilities = dt_model.predict_proba(input_df)[0]
        churn_prob = probabilities[1] * 100  

        col_gauge, col_text = st.columns([1, 1])
        
        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = churn_prob,
                title = {'text': "Churn Probability Risk (%)", 'font': {'color': "#1E3A8A", 'size': 18}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#1E3A8A"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, 30], 'color': "#F1F5F9"},
                        {'range': [30, 60], 'color': "#CBD5E1"},
                        {'range': [60, 100], 'color': "#94A3B8"}],
                    'threshold': {'line': {'color': "#EF4444", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col_text:
            st.markdown("### System Recommendation")
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown('<div class="risk-alert"><strong>🚨 CRITICAL RISK:</strong> This customer profile exhibits a high probability of churn. Immediate retention strategies (e.g., promotional discounts, loyalty outreach) are highly recommended.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safe-alert"><strong>✅ STABLE ACCOUNT:</strong> This customer is currently classified as loyal. No immediate retention intervention is required at this time.</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Model Confidence:** The algorithm is **{max(probabilities)*100:.1f}%** confident in this specific prediction based on the learned dataset patterns.")

with tab2:
    with st.container(border=True):
        st.header("AI Customer Segmentation")
        
        cluster_id = kmeans_model.predict(scaled_nums)[0]
        
        clusters = {
            0: {"name": "Moderate Spend / Budget", "desc": "Customers who maintain average billing over a moderate period."},
            1: {"name": "High Loyalty / Premium VIP", "desc": "The most valuable cohort. High lifetime value and long-term retention."},
            2: {"name": "Low Spending / Short Tenure", "desc": "New or budget-conscious customers with minimal financial commitment."}
        }
        
        st.subheader(f"Assigned Cohort: Cluster {cluster_id} - {clusters[cluster_id]['name']}")
        st.write(f"**Profile Description:** {clusters[cluster_id]['desc']}")
        
        cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
        
        fig2 = go.Figure(data=[
            go.Bar(name='This Customer', x=['Tenure (Months)', 'Monthly Charge ($)'], y=[tenure, monthly_charges], marker_color='#1E3A8A'),
            go.Bar(name='VIP Average (Cluster 1)', x=['Tenure (Months)', 'Monthly Charge ($)'], y=[cluster_centers[1][0], cluster_centers[1][1]], marker_color='#94A3B8')
        ])
        fig2.update_layout(
            barmode='group', 
            title="Customer vs. Premium VIP Averages",
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#1E3A8A")
        )
        st.plotly_chart(fig2, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Built with scikit-learn, Streamlit & Plotly | 2026")