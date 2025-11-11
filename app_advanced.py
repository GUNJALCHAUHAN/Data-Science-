import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import DBSCAN
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import folium
from streamlit_folium import folium_static
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Advanced Water Level Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
    }
    .highlight {
        background-color: #e6f3ff;
        padding: 20px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('cgwb-changes-in-depth-to-water-level.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['season'] = pd.cut(df['month'], 
                         bins=[0, 2, 5, 9, 12], 
                         labels=['Winter', 'Summer', 'Monsoon', 'Post-Monsoon'])
    
    # Calculate additional features with proper index alignment
    df = df.sort_values(['station_name', 'date'])  # Sort data first
    
    # Calculate rate of change
    df['rate_of_change'] = df.groupby('station_name')['level_diff'].diff()
    
    # Calculate cumulative change
    df['cumulative_change'] = df.groupby('station_name')['level_diff'].cumsum()
    
    # Calculate volatility using rolling window
    volatility = df.groupby('station_name')['level_diff'].rolling(window=3).std()
    volatility = volatility.reset_index(level=0, drop=True)  # Reset the multi-index
    df['volatility'] = volatility.fillna(0)
    
    return df

# Load data
df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["🎯 Executive Summary",
     "📊 Advanced Analytics",
     "🌍 Geospatial Insights",
     "🤖 Advanced ML Models",
     "📈 Time Series Analysis"])

if page == "🎯 Executive Summary":
    st.title("Executive Summary: Water Level Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Observations", f"{len(df):,}")
        
    with col2:
        st.metric("Date Range", f"{df['date'].min().year} - {df['date'].max().year}")
        
    with col3:
        st.metric("Stations Monitored", f"{df['station_name'].nunique():,}")
    
    # Key Insights Section
    st.subheader("Key Insights")
    
    # Calculate insights
    critical_stations = df[df['level_diff'] < df['level_diff'].quantile(0.1)]['station_name'].unique()
    recovery_stations = df[df['level_diff'] > df['level_diff'].quantile(0.9)]['station_name'].unique()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"📉 {len(critical_stations)} stations showing critical decline")
        
    with col2:
        st.success(f"📈 {len(recovery_stations)} stations showing significant recovery")
    
    # Trend Analysis
    st.subheader("Trend Analysis")
    trend_data = df.groupby('year')['level_diff'].agg(['mean', 'std']).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_data['year'],
        y=trend_data['mean'],
        mode='lines+markers',
        name='Mean Level Difference',
        error_y=dict(type='data', array=trend_data['std'], visible=True)
    ))
    fig.update_layout(title='Annual Water Level Trends with Uncertainty',
                     xaxis_title='Year',
                     yaxis_title='Level Difference (m)')
    st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Advanced Analytics":
    st.title("Advanced Analytics Dashboard")
    
    # Statistical Analysis
    st.subheader("Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate autocorrelation
        station_choice = st.selectbox("Select Station for Analysis", df['station_name'].unique())
        station_data = df[df['station_name'] == station_choice]['level_diff']
        
        if len(station_data) > 1:
            autocorr = pd.Series(station_data).autocorr()
            st.metric("Temporal Autocorrelation", f"{autocorr:.3f}")
            
            # Calculate trend significance
            x = range(len(station_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, station_data)
            st.metric("Trend Significance (p-value)", f"{p_value:.3f}")
    
    with col2:
        # Seasonal decomposition visualization
        if len(station_data) > 2:
            try:
                decomposition = seasonal_decompose(station_data, period=12, model='additive')
                fig = make_subplots(rows=3, cols=1)
                fig.add_trace(go.Scatter(y=decomposition.trend, name="Trend"), row=1, col=1)
                fig.add_trace(go.Scatter(y=decomposition.seasonal, name="Seasonal"), row=2, col=1)
                fig.add_trace(go.Scatter(y=decomposition.resid, name="Residual"), row=3, col=1)
                fig.update_layout(height=600, title_text="Time Series Decomposition")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Insufficient data for seasonal decomposition")

    # Advanced Pattern Analysis
    st.subheader("Pattern Analysis")
    
    # Calculate water stress indicators
    df['stress_indicator'] = (df['level_diff'] < df['level_diff'].quantile(0.25)).astype(int)
    stress_by_season = df.groupby(['season', 'state_name'])['stress_indicator'].mean().reset_index()
    
    fig = px.treemap(stress_by_season, 
                     path=['season', 'state_name'],
                     values='stress_indicator',
                     color='stress_indicator',
                     title='Water Stress Patterns by Region and Season')
    st.plotly_chart(fig, use_container_width=True)

elif page == "🌍 Geospatial Insights":
    st.title("Geospatial Insights Dashboard")
    
    try:
        # Create a base map centered on India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        
        # Clustering Analysis
        st.subheader("Spatial Clustering Analysis")
        
        # Prepare data for clustering
        coords = df[['latitude', 'longitude', 'level_diff']].dropna()
        
        if not coords.empty:
            # Standardize coordinates for clustering
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords[['latitude', 'longitude']])
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(coords_scaled)
            coords['cluster'] = clustering.labels_
            
            # Add clusters to map (limit points for performance)
            sample_size = min(5000, len(coords))
            sampled_coords = coords.sample(n=sample_size, random_state=42)
            
            for idx, row in sampled_coords.iterrows():
                try:
                    color = 'red' if row['level_diff'] < 0 else 'green'
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        color=color,
                        fill=True,
                        popup=f"Level Diff: {row['level_diff']:.2f}m"
                    ).add_to(m)
                except Exception as e:
                    st.warning(f"Skipping invalid coordinate point: {e}")
            
            # Display the map
            folium_static(m)
            
            # Spatial Statistics
            st.subheader("Spatial Statistics")
            
            state_stats = df.groupby('state_name').agg({
                'level_diff': ['mean', 'std', 'count'],
                'latitude': 'mean',
                'longitude': 'mean'
            }).reset_index()
            
            state_stats.columns = ['state_name', 'mean', 'std', 'count', 'latitude', 'longitude']
            
            # Create scatter map
            fig = px.scatter_mapbox(
                state_stats,
                lat='latitude',
                lon='longitude',
                size='count',
                color='mean',
                hover_name='state_name',
                hover_data=['std', 'count'],
                title='Spatial Distribution of Water Level Changes',
                color_continuous_scale='RdYlBu',
                size_max=50
            )
            
            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox=dict(
                    center=dict(lat=20.5937, lon=78.9629),
                    zoom=4
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional Statistics
            st.subheader("Regional Statistics")
            stats_df = state_stats.sort_values('mean', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Top 5 States with Rising Water Levels")
                st.dataframe(stats_df.head()[['state_name', 'mean', 'std']].round(3))
            
            with col2:
                st.markdown("### Top 5 States with Declining Water Levels")
                st.dataframe(stats_df.tail()[['state_name', 'mean', 'std']].round(3))
        
        else:
            st.warning("No valid coordinate data available for mapping.")
    
    except Exception as e:
        st.error(f"Error in geospatial analysis: {str(e)}")
        st.error("Please check if the latitude and longitude data is valid.")

elif page == "🤖 Advanced ML Models":
    st.title("Advanced Machine Learning Models")
    
    # Data Preparation
    st.subheader("Model Training and Evaluation")
    
    # Feature Engineering with progress indicator
    with st.spinner("Preparing features..."):
        # Enhanced feature set
        features = [
            'latitude', 'longitude', 'year', 'month', 
            'rate_of_change', 'cumulative_change', 'volatility'
        ]
        
        # Add encoded seasonal features
        df_model = pd.get_dummies(df['season'], prefix='season')
        for col in df_model.columns:
            df[col] = df_model[col]
        features.extend(df_model.columns.tolist())
        
        # Create the feature matrix
        X = df[features].copy()
        y = df['level_diff'].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Split the data with proper shuffling
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=True
        )
    
    # Model Selection with enhanced options
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox(
            "Select Model",
            ["Random Forest", "XGBoost", "LSTM", "Ensemble"]
        )
    
    with col2:
        # Add hyperparameter tuning options
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of Trees", 50, 200, 100, 25)
            max_depth = st.slider("Maximum Depth", 5, 30, 15, 5)
        elif model_choice == "XGBoost":
            learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2])
            max_depth = st.slider("Maximum Depth", 3, 10, 6)
        elif model_choice == "LSTM":
            epochs = st.slider("Number of Epochs", 5, 50, 10)
            batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128])
    
    if st.button("Train Model"):
        with st.spinner(f"Training {model_choice} model..."):
            try:
                if model_choice == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Feature importance plot
                    importance_df = pd.DataFrame({
                        'feature': features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        title='Feature Importance',
                        orientation='h'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                elif model_choice == "XGBoost":
                    model = xgb.XGBRegressor(
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                elif model_choice == "LSTM":
                    # Reshape data for LSTM
                    X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
                    X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
                    
                    model = Sequential([
                        LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
                        Dropout(0.2),
                        Dense(25, activation='relu'),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    
                    # Training progress bar
                    with st.empty():
                        history = model.fit(
                            X_train_reshaped, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2,
                            verbose=0
                        )
                        
                        # Plot training history
                        fig_history = go.Figure()
                        fig_history.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                        fig_history.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                        fig_history.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss')
                        st.plotly_chart(fig_history, use_container_width=True)
                        
                    y_pred = model.predict(X_test_reshaped).flatten()
                    
                else:  # Ensemble
                    models = [
                        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                        GradientBoostingRegressor(random_state=42),
                        xgb.XGBRegressor(random_state=42, n_jobs=-1)
                    ]
                    
                    predictions = []
                    for m in models:
                        m.fit(X_train, y_train)
                        predictions.append(m.predict(X_test))
                    
                    y_pred = np.mean(predictions, axis=0)
                
                # Model Evaluation
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = np.mean(np.abs(y_test - y_pred))
                
                # Display metrics in a better format
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MSE", f"{mse:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col3:
                    st.metric("R² Score", f"{r2:.4f}")
                with col4:
                    st.metric("MAE", f"{mae:.4f}")
                
                # Enhanced visualization of results
                fig = make_subplots(rows=2, cols=1, subplot_titles=('Predictions vs Actual', 'Residuals'))
                
                # Predictions vs Actual
                fig.add_trace(
                    go.Scatter(x=range(len(y_test)), y=y_test, name="Actual", mode='lines'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=range(len(y_pred)), y=y_pred, name="Predicted", mode='lines'),
                    row=1, col=1
                )
                
                # Residuals plot
                residuals = y_test - y_pred
                fig.add_trace(
                    go.Scatter(x=range(len(residuals)), y=residuals, mode='markers', name="Residuals"),
                    row=2, col=1
                )
                
                fig.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional analysis
                st.subheader("Error Distribution")
                fig_hist = px.histogram(residuals, nbins=50, title="Distribution of Prediction Errors")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred during model training: {str(e)}")
                st.error("Please try with different parameters or a different model.")

else:  # Time Series Analysis
    st.title("Time Series Analysis Dashboard")
    
    # Time Series Decomposition
    st.subheader("Temporal Pattern Analysis")
    
    # Select station for analysis
    station = st.selectbox("Select Station", df['station_name'].unique())
    station_data = df[df['station_name'] == station].sort_values('date')
    
    if len(station_data) > 0:
        # Create time series plot
        fig = make_subplots(rows=2, cols=1)
        
        # Water level trend
        fig.add_trace(
            go.Scatter(x=station_data['date'], y=station_data['currentlevel'],
                      name="Water Level"),
            row=1, col=1
        )
        
        # Rate of change
        fig.add_trace(
            go.Scatter(x=station_data['date'], y=station_data['rate_of_change'],
                      name="Rate of Change"),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text=f"Time Series Analysis for {station}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal Analysis
        seasonal_stats = station_data.groupby('season')['level_diff'].agg(['mean', 'std']).reset_index()
        
        fig = go.Figure(data=[
            go.Bar(name='Mean Change', x=seasonal_stats['season'], y=seasonal_stats['mean'],
                  error_y=dict(type='data', array=seasonal_stats['std']))
        ])
        fig.update_layout(title="Seasonal Water Level Changes",
                         xaxis_title="Season",
                         yaxis_title="Level Difference (m)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly Detection
        st.subheader("Anomaly Detection")
        
        # Calculate z-scores for level changes safely
        level_diff_clean = station_data['level_diff'].fillna(station_data['level_diff'].mean())
        z_scores = np.abs((level_diff_clean - level_diff_clean.mean()) / level_diff_clean.std())
        anomalies = station_data[z_scores > 2]  # Points beyond 2 standard deviations
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=station_data['date'], y=station_data['level_diff'],
                               mode='lines', name='Normal'))
        fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['level_diff'],
                               mode='markers', name='Anomaly',
                               marker=dict(size=10, color='red')))
        fig.update_layout(title="Anomaly Detection in Water Level Changes",
                         xaxis_title="Date",
                         yaxis_title="Level Difference (m)")
        st.plotly_chart(fig, use_container_width=True)
        
        if len(anomalies) > 0:
            st.warning(f"Detected {len(anomalies)} anomalous measurements")
            st.dataframe(anomalies[['date', 'level_diff']])