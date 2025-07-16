import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from utils import CarbonAnalyzer, ChartGenerator, CarbonTips, format_number

# Set page configuration
st.set_page_config(
    page_title="Carbon Tracker",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4A90E2;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background-color: #f0f8ff;
        border-left: 4px solid #4A90E2;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .calculator-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        # Main dataset
        df = pd.read_csv("data source/india_carbon_dataset.csv")
        
        # Fuel data
        fuel_df = pd.read_csv("data source/International Energy Agency - CO2 emissions by fuel, India, 2022.csv")
        fuel_historical = pd.read_csv("data source/International Energy Agency - CO2 emissions by fuel in India.csv")
        
        # Sector data
        sector_df = pd.read_csv("data source/International Energy Agency - CO2 emissions by sector, India, 2022.csv")
        sector_historical = pd.read_csv("data source/International Energy Agency - CO2 emissions by sector in India.csv")
        
        # Per capita data
        per_capita_df = pd.read_csv("data source/International Energy Agency - CO2 emissions per capita, Asia Pacific.csv")
        
        return df, fuel_df, fuel_historical, sector_df, sector_historical, per_capita_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None

@st.cache_resource
def train_model(df):
    """Train the Random Forest model"""
    try:
        features = ['temperature', 'travel_km', 'electricity_units', 'food_impact',
                   'waste_kg', 'construction_score', 'agriculture_score', 'lifestyle_score']
        target = 'co2_emission_kg'
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, mse, r2, X.columns
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None

def create_fuel_chart(fuel_df):
    """Create fuel emissions chart"""
    fuel_summary = fuel_df.groupby("CO2 emissions by fuel, India, 2022")['Value'].sum().reset_index()
    fuel_summary['percentage'] = 100 * fuel_summary['Value'] / fuel_summary['Value'].sum()
    fuel_summary = fuel_summary.sort_values(by='percentage', ascending=False)
    
    fig = px.pie(fuel_summary, 
                 values='percentage', 
                 names='CO2 emissions by fuel, India, 2022',
                 title='CO₂ Emissions by Fuel Type (2022)',
                 color_discrete_sequence=['#FF6F3C', '#FFBD3C', '#3CC9F0'])
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500, font_size=12)
    
    return fig

def create_historical_fuel_chart(fuel_historical):
    """Create historical fuel emissions chart"""
    fuel_historical.columns = ['Fuel', 'Value', 'Year', 'Units']
    fuel_historical['Year'] = pd.to_numeric(fuel_historical['Year'], errors='coerce')
    fuel_historical = fuel_historical.dropna()
    
    fig = px.line(fuel_historical, 
                  x='Year', 
                  y='Value', 
                  color='Fuel',
                  title='Evolution of CO₂ Emissions by Fuel (2000-2022)',
                  labels={'Value': 'CO₂ Emissions (Mt)', 'Year': 'Year'})
    
    fig.update_layout(height=500, hovermode='x unified')
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    
    return fig

def create_sector_chart(sector_df):
    """Create sector emissions chart"""
    sector_df = sector_df.iloc[:, :2]
    sector_df.columns = ['Sector', 'Value']
    sector_df = sector_df.dropna()
    sector_df['Value'] = pd.to_numeric(sector_df['Value'], errors='coerce')
    sector_df['Percentage'] = 100 * sector_df['Value'] / sector_df['Value'].sum()
    
    fig = px.bar(sector_df, 
                 x='Percentage', 
                 y='Sector', 
                 orientation='h',
                 title='CO₂ Emissions by Sector (2022)',
                 labels={'Percentage': 'Percentage (%)', 'Sector': 'Sector'},
                 color='Percentage',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def create_feature_importance_chart(model, feature_names):
    """Create feature importance chart"""
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(feature_df, 
                 x='Importance', 
                 y='Feature', 
                 orientation='h',
                 title='Feature Importance in CO₂ Prediction',
                 color='Importance',
                 color_continuous_scale='RdYlBu_r')
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def calculate_carbon_footprint(model, temperature, travel_km, electricity_units, 
                             food_impact, waste_kg, construction_score, 
                             agriculture_score, lifestyle_score):
    """Calculate carbon footprint using the trained model"""
    input_data = pd.DataFrame([[
        temperature, travel_km, electricity_units, food_impact,
        waste_kg, construction_score, agriculture_score, lifestyle_score
    ]], columns=[
        'temperature', 'travel_km', 'electricity_units', 'food_impact',
        'waste_kg', 'construction_score', 'agriculture_score', 'lifestyle_score'
    ])
    
    prediction = model.predict(input_data)[0]
    return prediction

def main():
    # Header
    st.markdown('<h1 class="main-header">🌱 Carbon Tracker Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">A comprehensive platform for analyzing CO₂ emissions and calculating your carbon footprint</div>', unsafe_allow_html=True)
    
    # Load data
    df, fuel_df, fuel_historical, sector_df, sector_historical, per_capita_df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Train model
    model, mse, r2, feature_names = train_model(df)
    
    if model is None:
        st.error("Failed to train model.")
        return
    
    # Calculate average footprint for comparison (used in multiple sections)
    avg_footprint = df['co2_emission_kg'].mean()
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Dashboard", "🧮 Carbon Calculator", "📈 Data Analysis", "🤖 Model Performance"]
    )
    
    if page == "📊 Dashboard":
        st.markdown('<h2 class="sub-header">📊 CO₂ Emissions Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_emissions = fuel_df['Value'].sum()
            st.metric("Total Emissions (2022)", f"{total_emissions:.1f} Mt", "📈")
        
        with col2:
            coal_percentage = (fuel_df[fuel_df['CO2 emissions by fuel, India, 2022'] == 'Coal']['Value'].sum() / total_emissions) * 100
            st.metric("Coal Share", f"{coal_percentage:.1f}%", "🔥")
        
        with col3:
            st.metric("Avg. Personal Footprint", f"{avg_footprint:.1f} kg", "👤")
        
        with col4:
            model_accuracy = r2 * 100
            st.metric("Model Accuracy", f"{model_accuracy:.1f}%", "🎯")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fuel_chart = create_fuel_chart(fuel_df)
            st.plotly_chart(fuel_chart, use_container_width=True)
        
        with col2:
            sector_chart = create_sector_chart(sector_df)
            st.plotly_chart(sector_chart, use_container_width=True)
        
        # Historical trend
        st.markdown('<h3 class="sub-header">📈 Historical Trends</h3>', unsafe_allow_html=True)
        historical_chart = create_historical_fuel_chart(fuel_historical)
        st.plotly_chart(historical_chart, use_container_width=True)
    
    elif page == "🧮 Carbon Calculator":
        st.markdown('<h2 class="sub-header">🧮 Personal Carbon Footprint Calculator</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="calculator-section">
            <h3>Calculate Your Carbon Footprint</h3>
            <p>Enter your daily activities and lifestyle choices to estimate your CO₂ emissions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input form
        with st.form("carbon_calculator"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌡️ Environmental & Travel")
                temperature = st.slider("Average Temperature (°C)", 0, 50, 25)
                travel_km = st.slider("Daily Travel Distance (km)", 0, 500, 50)
                electricity_units = st.slider("Monthly Electricity Usage (units)", 0, 1000, 250)
                food_impact = st.slider("Food Impact Score (1-10)", 1, 10, 5)
            
            with col2:
                st.subheader("🏠 Lifestyle & Consumption")
                waste_kg = st.slider("Daily Waste Generated (kg)", 0, 50, 5)
                construction_score = st.slider("Construction Impact Score (1-10)", 1, 10, 3)
                agriculture_score = st.slider("Agriculture Impact Score (1-10)", 1, 10, 2)
                lifestyle_score = st.slider("General Lifestyle Score (1-10)", 1, 10, 5)
            
            submitted = st.form_submit_button("Calculate Carbon Footprint 🌱")
            
            if submitted:
                footprint = calculate_carbon_footprint(
                    model, temperature, travel_km, electricity_units,
                    food_impact, waste_kg, construction_score,
                    agriculture_score, lifestyle_score
                )
                
                st.success(f"Your estimated carbon footprint: **{footprint:.2f} kg CO₂**")
                
                # Comparison
                if footprint < avg_footprint:
                    st.info("🎉 Great! Your footprint is below average.")
                elif footprint > avg_footprint * 1.5:
                    st.warning("⚠️ Your footprint is significantly above average. Consider reducing your impact!")
                else:
                    st.info("📊 Your footprint is around average.")
                
                # Tips
                st.markdown("### 💡 Tips to Reduce Your Carbon Footprint:")
                tips = [
                    "🚲 Use public transport, walk, or bike instead of driving",
                    "💡 Switch to LED bulbs and energy-efficient appliances",
                    "🌱 Eat more plant-based meals and less meat",
                    "♻️ Reduce, reuse, and recycle waste",
                    "🌡️ Use programmable thermostats and insulate your home",
                    "💧 Conserve water and fix leaks promptly"
                ]
                for tip in tips:
                    st.markdown(f"• {tip}")
    
    elif page == "📈 Data Analysis":
        st.markdown('<h2 class="sub-header">📈 Data Analysis & Insights</h2>', unsafe_allow_html=True)
        
        # Data overview
        st.subheader("📊 Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Dataset Size:** {df.shape[0]} records, {df.shape[1]} features")
            st.info(f"**Date Range:** {df['date'].min()} to {df['date'].max()}")
        
        with col2:
            categories = df['category'].value_counts()
            st.info(f"**Categories:** {', '.join(categories.index)}")
            st.info(f"**States Covered:** {df['state'].nunique()}")
        
        # Distribution plots
        st.subheader("📊 Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CO2 emissions distribution
            fig = px.histogram(df, x='co2_emission_kg', nbins=20, 
                             title='Distribution of CO₂ Emissions')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Emissions by category
            fig = px.box(df, x='category', y='co2_emission_kg',
                        title='CO₂ Emissions by Category')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("🔗 Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # State-wise analysis
        st.subheader("🗺️ State-wise Emissions")
        state_emissions = df.groupby('state')['co2_emission_kg'].agg(['mean', 'sum', 'count']).reset_index()
        state_emissions.columns = ['State', 'Average', 'Total', 'Count']
        
        fig = px.bar(state_emissions, x='State', y='Average',
                    title='Average CO₂ Emissions by State')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Model Performance":
        st.markdown('<h2 class="sub-header">🤖 Machine Learning Model Performance</h2>', unsafe_allow_html=True)
        
        # Model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Squared Error", f"{mse:.2f}")
        
        with col2:
            st.metric("R² Score", f"{r2:.3f}")
        
        with col3:
            st.metric("Model Type", "Random Forest")
        
        # Feature importance
        st.subheader("📊 Feature Importance")
        importance_chart = create_feature_importance_chart(model, feature_names)
        st.plotly_chart(importance_chart, use_container_width=True)
        
        # Model details
        st.subheader("⚙️ Model Configuration")
        st.json({
            "algorithm": "Random Forest Regressor",
            "n_estimators": 100,
            "random_state": 42,
            "test_size": 0.2,
            "features": list(feature_names)
        })
        
        # Predictions vs Actual
        st.subheader("📈 Model Predictions")
        
        # Get predictions for visualization
        features = ['temperature', 'travel_km', 'electricity_units', 'food_impact',
                   'waste_kg', 'construction_score', 'agriculture_score', 'lifestyle_score']
        X = df[features]
        y = df['co2_emission_kg']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        
        # Create scatter plot
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={'x': 'Actual CO₂ Emissions', 'y': 'Predicted CO₂ Emissions'},
                        title='Actual vs Predicted CO₂ Emissions')
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🌱 Carbon Tracker - Making environmental impact visible and actionable</p>
        <p>Built with ❤️ using Streamlit | Data source: International Energy Agency</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
