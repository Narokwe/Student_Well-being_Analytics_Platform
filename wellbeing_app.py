import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Youth First Kenya - Student Well-being Analytics",
    page_icon=":school:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('student_wellbeing_data.csv')

@st.cache_data
def load_model():
    model_data = joblib.load('wellbeing_model.pkl')
    return model_data['model'], model_data['expected_columns']

df = load_data()
model, expected_columns = load_model()

# Convert survey_date to datetime
df['survey_date'] = pd.to_datetime(df['survey_date'])

# Sidebar filters
st.sidebar.header("Filters")
selected_region = st.sidebar.multiselect(
    "Select Region(s)",
    options=df['region'].unique(),
    default=df['region'].unique()
)

selected_school = st.sidebar.multiselect(
    "Select School(s)",
    options=df['school_id'].unique(),
    default=[]
)

selected_gender = st.sidebar.multiselect(
    "Select Gender(s)",
    options=df['gender'].unique(),
    default=df['gender'].unique()
)

age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(int(df['age'].min()), int(df['age'].max()))
)

# Apply filters
filtered_df = df[
    (df['region'].isin(selected_region)) &
    (df['gender'].isin(selected_gender)) &
    (df['age'] >= age_range[0]) &
    (df['age'] <= age_range[1])
]

if selected_school:
    filtered_df = filtered_df[filtered_df['school_id'].isin(selected_school)]

# Main app
st.title("Youth First Kenya - Student Well-being Analytics Platform")
st.markdown("""
This platform provides insights into student well-being metrics and predicts which students 
may be at risk of poor mental health outcomes.
""")

# Dashboard metrics
st.header("Overview Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", len(filtered_df))
col2.metric("Average Resilience Score", f"{filtered_df['resilience_score'].mean():.1f}")
col3.metric("High Risk Students", f"{filtered_df['risk_classification'].sum()} ({(filtered_df['risk_classification'].sum() / len(filtered_df)) * 100:.1f}%)")
col4.metric("Intervention Coverage", f"{(filtered_df['intervention_status'].sum() / len(filtered_df)) * 100:.1f}%")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "Trend Analysis", 
    "Demographic Insights", 
    "Risk Prediction", 
    "Data Explorer"
])

with tab1:
    st.header("Trend Analysis")
    
    # Time series analysis
    time_df = filtered_df.set_index('survey_date')
    monthly_avg = time_df.resample('M').mean(numeric_only=True)
    
    fig = px.line(monthly_avg, x=monthly_avg.index, y=['resilience_score', 'stress_levels'],
                 title='Monthly Average Resilience Score and Stress Levels',
                 labels={'value': 'Score', 'variable': 'Metric'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Metric trends
    st.subheader("Well-being Metrics Over Time")
    selected_metrics = st.multiselect(
        "Select metrics to compare",
        options=['growth_mindset', 'self_esteem', 'stress_levels', 'social_support', 'academic_pressure'],
        default=['growth_mindset', 'self_esteem']
    )
    
    if selected_metrics:
        fig = px.line(monthly_avg, x=monthly_avg.index, y=selected_metrics,
                     title='Monthly Average of Selected Metrics',
                     labels={'value': 'Score', 'variable': 'Metric'})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Demographic Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Resilience by region
        fig = px.box(filtered_df, x='region', y='resilience_score',
                    title='Resilience Score Distribution by Region')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Resilience by gender
        fig = px.box(filtered_df, x='gender', y='resilience_score',
                    title='Resilience Score Distribution by Gender')
        st.plotly_chart(fig, use_container_width=True)
    
    # Age vs Resilience
    fig = px.scatter(filtered_df, x='age', y='resilience_score', color='gender',
                    trendline="lowess",
                    title='Age vs Resilience Score')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Risk Prediction Module")
    st.markdown("""
    This module predicts which students are at highest risk of poor mental well-being 
    based on their survey responses.
    """)
    
    # Prediction interface
    st.subheader("Predict Risk for New Students")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 13, 19, 15)
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
            growth_mindset = st.slider("Growth Mindset (1-10)", 1, 10, 6)
            self_esteem = st.slider("Self Esteem (1-10)", 1, 10, 6)
        
        with col2:
            stress_levels = st.slider("Stress Levels (1-10)", 1, 10, 5)
            social_support = st.slider("Social Support (1-10)", 1, 10, 7)
            academic_pressure = st.slider("Academic Pressure (1-10)", 1, 10, 6)
        
        submitted = st.form_submit_button("Predict Risk")
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'growth_mindset': [growth_mindset],
                'self_esteem': [self_esteem],
                'stress_levels': [stress_levels],
                'social_support': [social_support],
                'academic_pressure': [academic_pressure],
                'age': [age],
                'gender': [gender]
            })
            
            # One-hot encode
            input_data = pd.get_dummies(input_data)
            # Ensure all expected columns are present
            for col in expected_columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Reorder columns
            input_data = input_data[expected_columns]
            
            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            # Display results
            if prediction[0] == 1:
                st.error(f"High Risk Predicted ({prediction_proba[0][1]*100:.1f}% probability)")
                st.markdown("""
                **Recommended Actions:**
                - Prioritize for Youth First intervention
                - Consider additional counseling
                - Monitor closely
                """)
            else:
                st.success(f"Low Risk Predicted ({prediction_proba[0][0]*100:.1f}% probability)")
                st.markdown("""
                **Recommended Actions:**
                - Continue with standard support
                - Monitor in next survey cycle
                """)
    
    # High risk students in current data
    st.subheader("Identified High Risk Students in Current Data")
    high_risk_df = filtered_df[filtered_df['risk_classification'] == 1].sort_values('resilience_score')
    st.dataframe(high_risk_df.head(20), use_container_width=True)
    
    if not high_risk_df.empty:
        # Download button
        st.download_button(
            label="Download High Risk Students Data",
            data=high_risk_df.to_csv(index=False),
            file_name='high_risk_students.csv',
            mime='text/csv'
        )

with tab4:
    st.header("Data Explorer")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download filtered data
    st.download_button(
        label="Download Filtered Data",
        data=filtered_df.to_csv(index=False),
        file_name='filtered_student_data.csv',
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.markdown("""
**Youth First Kenya** - Building resilience in young people through evidence-based interventions.
""")