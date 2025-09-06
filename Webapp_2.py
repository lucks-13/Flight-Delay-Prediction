import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Flight Delay Prediction WebApp", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main > div {
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        background: transparent;
        border-radius: 8px;
    }
    .stMetric {
        background: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    .plot-container {
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .prediction-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 120px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_data():
    try:
        with open('model_for_Web2.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please run code.ipynb first for model_for_Web2.pkl.")
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
    st.title("Flight Delay Prediction")
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
    
    tab1, tab2, tab3 = st.tabs(["Predict", "Analytics", "Models"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Flight Info")
            airline = st.selectbox("Airline", airlines, key="airline")
            origin = st.selectbox("Origin", airports, key="origin")
            destination = st.selectbox("Destination", airports, key="dest")
            distance = st.slider("Distance (miles)", 100, 3000, 1000, key="dist")
            scheduled_departure = st.time_input("Departure Time", value=pd.Timestamp("12:00").time(), key="time")
            
            departure_minutes = scheduled_departure.hour * 100 + scheduled_departure.minute
            
        with col2:
            st.subheader("Date & Conditions")
            day_of_week = st.selectbox("Day", [1, 2, 3, 4, 5, 6, 7], 
                                     format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x-1], key="dow")
            month = st.selectbox("Month", list(range(1, 13)), 
                               format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1], key="month")
            weather_delay = st.slider("Weather Impact", 0, 120, 0, key="weather")
            
            st.subheader("Models")
            model_cols = st.columns(2)
            selected_models = []
            for i, model_name in enumerate(models.keys()):
                with model_cols[i % 2]:
                    if st.checkbox(model_name, value=True, key=f"model_{i}"):
                        selected_models.append(model_name)
        
        if st.button("Start Prediction", type="primary", use_container_width=True):
            if selected_models:
                input_data = {
                    'airline': airline,
                    'origin': origin,
                    'destination': destination,
                    'distance': distance,
                    'scheduled_departure': departure_minutes,
                    'day_of_week': day_of_week,
                    'month': month,
                    'weather_delay': weather_delay
                }
                
                processed_input = preprocess_input(input_data, encoders, scaler, feature_columns, categorical_columns)
                
                st.subheader("Predictions")
                predictions = {}
                
                for model_name in selected_models:
                    model = models[model_name]
                    if 'Linear' in model_name or 'Ridge' in model_name or 'Lasso' in model_name:
                        processed_input_scaled = scaler.transform(processed_input)
                        pred = model.predict(processed_input_scaled)[0]
                    else:
                        pred = model.predict(processed_input)[0]
                    predictions[model_name] = max(0, pred)
                
                num_models = len(predictions)
                cols_per_row = 5
                
                for i in range(0, num_models, cols_per_row):
                    row_predictions = list(predictions.items())[i:i+cols_per_row]
                    cols = st.columns(len(row_predictions))
                    
                    for j, (model_name, pred) in enumerate(row_predictions):
                        r2_score = performance[model_name]['r2']
                        with cols[j]:
                            color = "normal" if pred < 15 else "off" if pred > 60 else "normal"
                            st.metric(
                                f"{model_name.split()[0]}", 
                                f"{pred:.0f}min", 
                                f"R²={r2_score:.2f}"
                            )
                
                avg_pred = np.mean(list(predictions.values()))
                if avg_pred < 15:
                    st.success(f"Result : Low delay expected: {avg_pred:.0f} minutes")
                elif avg_pred < 60:
                    st.warning(f"Result :  Moderate delay expected: {avg_pred:.0f} minutes")
                else:
                    st.error(f"Result :  High delay expected: {avg_pred:.0f} minutes")
            else:
                st.warning("Select at least one model")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig1 = px.histogram(original_data, x='delay_minutes', nbins=30, 
                              title='Delay Distribution', template='plotly_dark')
            fig1.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            airline_delays = original_data.groupby('airline')['delay_minutes'].mean().reset_index()
            fig2 = px.bar(airline_delays, x='airline', y='delay_minutes', 
                         title='Avg Delay by Airline', template='plotly_dark')
            fig2.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            original_data_copy = original_data.copy()
            original_data_copy['departure_hour'] = original_data_copy['scheduled_departure'] // 100
            hourly_delays = original_data_copy.groupby('departure_hour')['delay_minutes'].mean().reset_index()
            fig3 = px.line(hourly_delays, x='departure_hour', y='delay_minutes', 
                          title='Hourly Delay Pattern', template='plotly_dark')
            fig3.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            day_delays = original_data.groupby('day_of_week')['delay_minutes'].mean().reset_index()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_delays['day_name'] = day_delays['day_of_week'].apply(lambda x: day_names[x-1] if 1 <= x <= 7 else 'Unknown')
            fig4 = px.bar(day_delays, x='day_name', y='delay_minutes', 
                         title='Weekly Pattern', template='plotly_dark')
            fig4.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        sample_subset = original_data.sample(n=min(2000, len(original_data)), random_state=42)
        fig5 = px.scatter(sample_subset, x='distance', y='delay_minutes', 
                         title='Distance vs Delay', template='plotly_dark', opacity=0.6)
        fig5.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            delay_categories = pd.cut(original_data['delay_minutes'],
                                    bins=[0, 15, 30, 60, 120, float('inf')],
                                    labels=['On Time', 'Minor', 'Moderate', 'Major', 'Severe'])
            category_counts = delay_categories.value_counts()
            fig6 = px.pie(values=category_counts.values, names=category_counts.index, 
                         title='Delay Categories', template='plotly_dark')
            fig6.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig6, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            monthly_delays = original_data.groupby('month')['delay_minutes'].mean().reset_index()
            fig7 = px.line(monthly_delays, x='month', y='delay_minutes',
                          title='Monthly Trends', template='plotly_dark', markers=True)
            fig7.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig7, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        performance_df = pd.DataFrame(performance).T.reset_index()
        performance_df.columns = ['Model', 'MSE', 'R²']
        performance_df = performance_df.sort_values('R²', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig_r2 = px.bar(performance_df, x='R²', y='Model', orientation='h',
                           title='Model Performance (R²)', template='plotly_dark')
            fig_r2.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_r2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig_mse = px.bar(performance_df, x='MSE', y='Model', orientation='h',
                            title='Model Error (MSE)', template='plotly_dark')
            fig_mse.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_mse, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Performance Metrics")
        st.dataframe(performance_df, use_container_width=True, hide_index=True)
        
        if 'Random Forest' in models:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            rf_model = models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(feature_importance.head(10), x='Importance', y='Feature',
                                  orientation='h', title='Top 10 Feature Importances', template='plotly_dark')
            fig_importance.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_importance, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()