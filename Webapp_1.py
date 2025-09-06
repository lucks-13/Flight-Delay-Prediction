import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Flight Delay Prediction", layout="wide")

@st.cache_data
def load_model_data():
    try:
        with open('model_for_Web1.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please run code.ipynb first for model_for_Web1.pkl.")
        st.stop()

def preprocess_input(input_data, encoders, scaler, feature_columns, categorical_columns):
    input_df = pd.DataFrame([input_data])
    
    if 'scheduled_departure' in input_df.columns:
        input_df['departure_hour'] = input_df['scheduled_departure'] // 100
        input_df['departure_minute'] = input_df['scheduled_departure'] % 100
        input_df['is_rush_hour'] = input_df['departure_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        input_df['is_red_eye'] = input_df['departure_hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
        input_df['departure_time_sin'] = np.sin(2 * np.pi * input_df['departure_hour'] / 24)
        input_df['departure_time_cos'] = np.cos(2 * np.pi * input_df['departure_hour'] / 24)
    
    if 'month' in input_df.columns:
        input_df['is_summer'] = input_df['month'].isin([6, 7, 8]).astype(int)
        input_df['is_winter'] = input_df['month'].isin([12, 1, 2]).astype(int)
        input_df['is_holiday_season'] = input_df['month'].isin([11, 12]).astype(int)
        input_df['month_sin'] = np.sin(2 * np.pi * input_df['month'] / 12)
        input_df['month_cos'] = np.cos(2 * np.pi * input_df['month'] / 12)
    
    if 'day_of_week' in input_df.columns:
        input_df['is_weekend'] = input_df['day_of_week'].isin([6, 7]).astype(int)
        input_df['day_sin'] = np.sin(2 * np.pi * input_df['day_of_week'] / 7)
        input_df['day_cos'] = np.cos(2 * np.pi * input_df['day_of_week'] / 7)
    
    if 'distance' in input_df.columns:
        input_df['is_short_haul'] = (input_df['distance'] < 500).astype(int)
        input_df['distance_log'] = np.log1p(input_df['distance'])
    
    for col in categorical_columns:
        if col in input_df.columns and col in encoders:
            try:
                input_df[col] = encoders[col].transform(input_df[col])
            except ValueError:
                input_df[col] = 0
    
    input_features = input_df[feature_columns]
    return input_features

def main():
    st.title("Flight Delay Prediction Dashboard")
    st.caption("A Athena Award Project, Used all light weight high accuracy model derived from my analysis if you want highest accuracy use the below link for .pkl file")
    st.markdown("[Click Here For Source Code](https://github.com/lucks-13/Flight-Delay-Prediction)", unsafe_allow_html=True)
    
    model_data = load_model_data()
    models = model_data['models']
    scaler = model_data['scaler']
    encoders = model_data['encoders']
    feature_columns = model_data['feature_columns']
    performance = model_data['performance']
    sample_data = model_data['sample_data']
    categorical_columns = model_data['categorical_columns']
    airlines = model_data['airlines']
    airports = model_data['airports']
    original_data = model_data['original_data']
    
    tab1, tab2, tab3 = st.tabs(["Prediction Dashboard", "Data Analysis", "Model Performance"])
    
    with tab1:
        st.header("Flight Delay Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Flight Details")
            airline = st.selectbox("Airline", airlines)
            origin = st.selectbox("Origin Airport", airports)
            destination = st.selectbox("Destination Airport", airports)
            distance = st.slider("Distance (miles)", 100, 3000, 1000)
            scheduled_departure = st.slider("Scheduled Departure", 600, 2200, 1200)
            
        with col2:
            st.subheader("Date & Weather")
            day_of_week = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7], 
                                     format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x-1])
            month = st.selectbox("Month", list(range(1, 13)), 
                               format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])
            weather_delay = st.slider("Weather Delay (minutes)", 0, 120, 0)
            
            st.subheader("Model Selection")
            selected_models = []
            for model_name in models.keys():
                if st.checkbox(model_name, value=True):
                    selected_models.append(model_name)
        
        if st.button("Predict Delay"):
            if selected_models:
                input_data = {
                    'airline': airline,
                    'origin': origin,
                    'destination': destination,
                    'distance': distance,
                    'scheduled_departure': scheduled_departure,
                    'day_of_week': day_of_week,
                    'month': month,
                    'weather_delay': weather_delay
                }
                
                processed_input = preprocess_input(input_data, encoders, scaler, feature_columns, categorical_columns)
                
                st.subheader("Prediction Results")
                predictions = {}
                
                for model_name in selected_models:
                    model = models[model_name]
                    if model_name == 'Linear Regression':
                        processed_input_scaled = scaler.transform(processed_input)
                        pred = model.predict(processed_input_scaled)[0]
                    else:
                        pred = model.predict(processed_input)[0]
                    predictions[model_name] = max(0, pred)
                
                cols = st.columns(len(predictions))
                for i, (model_name, pred) in enumerate(predictions.items()):
                    r2_score = performance[model_name]['r2']
                    with cols[i]:
                        st.metric(f"{model_name}", f"{pred:.1f} min", f"R² = {r2_score:.3f}")
            else:
                st.warning("Please select at least one model.")
    
    with tab2:
        st.header("Flight Delay Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Delay Distribution")
            fig1 = px.histogram(original_data, x='delay_minutes', nbins=50, title='Delay Minutes Distribution')
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("Average Delay by Airline")
            airline_delays = original_data.groupby('airline')['delay_minutes'].mean().reset_index()
            fig2 = px.bar(airline_delays, x='airline', y='delay_minutes', title='Average Delay by Airline')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.subheader("Log-Transformed Delay")
            log_delays = np.log1p(original_data['delay_minutes'])
            fig3 = px.histogram(x=log_delays, nbins=50, title='Log-Transformed Delay Distribution')
            st.plotly_chart(fig3, use_container_width=True)
            
            st.subheader("Delay by Day of Week")
            day_delays = original_data.groupby('day_of_week')['delay_minutes'].mean().reset_index()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_delays['day_name'] = day_delays['day_of_week'].apply(lambda x: day_names[x-1] if 1 <= x <= 7 else 'Unknown')
            fig4 = px.bar(day_delays, x='day_name', y='delay_minutes', title='Average Delay by Day of Week')
            st.plotly_chart(fig4, use_container_width=True)
        
        st.subheader("Distance vs Delay Relationship")
        sample_subset = original_data.sample(n=min(3000, len(original_data)), random_state=42)
        fig5 = px.scatter(sample_subset, x='distance', y='delay_minutes', title='Distance vs Delay Minutes', opacity=0.6)
        st.plotly_chart(fig5, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Monthly Delay Patterns")
            monthly_delays = original_data.groupby('month')['delay_minutes'].agg(['mean', 'std']).reset_index()
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=monthly_delays['month'], y=monthly_delays['mean'],
                                    error_y=dict(type='data', array=monthly_delays['std']),
                                    mode='markers+lines', name='Average Delay'))
            fig6.update_layout(title='Monthly Delay Patterns', xaxis_title='Month', yaxis_title='Average Delay (min)')
            st.plotly_chart(fig6, use_container_width=True)
            
            st.subheader("Hourly Delay Pattern")
            original_data_copy = original_data.copy()
            original_data_copy['departure_hour'] = original_data_copy['scheduled_departure'] // 100
            hourly_delays = original_data_copy.groupby('departure_hour')['delay_minutes'].mean().reset_index()
            fig7 = px.line(hourly_delays, x='departure_hour', y='delay_minutes', title='Delay by Departure Hour')
            st.plotly_chart(fig7, use_container_width=True)
        
        with col4:
            st.subheader("Delay Severity Distribution")
            delay_categories = pd.cut(original_data['delay_minutes'],
                                    bins=[0, 15, 30, 60, 120, float('inf')],
                                    labels=['On Time', 'Minor', 'Moderate', 'Major', 'Severe'])
            category_counts = delay_categories.value_counts()
            fig8 = px.pie(values=category_counts.values, names=category_counts.index, 
                         title='Delay Severity Distribution')
            st.plotly_chart(fig8, use_container_width=True)
            
            st.subheader("Top Routes by Frequency")
            original_data_copy['route'] = original_data_copy['origin'] + '-' + original_data_copy['destination']
            top_routes = original_data_copy['route'].value_counts().head(10).reset_index()
            fig9 = px.bar(top_routes, x='count', y='route', orientation='h', title='Top 10 Routes')
            st.plotly_chart(fig9, use_container_width=True)
        
        st.subheader("Correlation Matrix")
        numerical_cols = sample_data.select_dtypes(include=[np.number]).columns
        corr_matrix = sample_data[numerical_cols].corr()
        fig10 = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
        st.plotly_chart(fig10, use_container_width=True)
    
    with tab3:
        st.header("Model Performance Comparison")
        
        performance_df = pd.DataFrame(performance).T
        performance_df = performance_df.reset_index()
        performance_df.columns = ['Model', 'MSE', 'R²']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("R² Score Comparison")
            fig_r2 = px.bar(performance_df, x='Model', y='R²', title='Model R² Scores')
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            st.subheader("MSE Comparison")
            fig_mse = px.bar(performance_df, x='Model', y='MSE', title='Model Mean Squared Error')
            st.plotly_chart(fig_mse, use_container_width=True)
        
        st.subheader("Model Performance Metrics")
        st.dataframe(performance_df, use_container_width=True)
        
        if 'Random Forest' in models:
            st.subheader("Feature Importance (Random Forest)")
            rf_model = models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(feature_importance.head(15), x='Importance', y='Feature',
                                  orientation='h', title='Top 15 Feature Importances')
            st.plotly_chart(fig_importance, use_container_width=True)
        
        st.subheader("Cumulative Distribution")
        sorted_delays = np.sort(original_data['delay_minutes'])
        y_vals = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
        fig_cum = px.line(x=sorted_delays, y=y_vals, title='Cumulative Distribution of Delays')
        fig_cum.update_layout(xaxis_title='Delay Minutes', yaxis_title='Cumulative Probability')
        st.plotly_chart(fig_cum, use_container_width=True)

if __name__ == "__main__":
    main()
