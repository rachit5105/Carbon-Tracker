import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class CarbonAnalyzer:
    """Main class for carbon emissions analysis and prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_metrics = {}
    
    def load_data(self):
        """Load all datasets"""
        data = {}
        try:
            data['main'] = pd.read_csv("data source/india_carbon_dataset.csv")
            data['fuel_2022'] = pd.read_csv("data source/International Energy Agency - CO2 emissions by fuel, India, 2022.csv")
            data['fuel_historical'] = pd.read_csv("data source/International Energy Agency - CO2 emissions by fuel in India.csv")
            data['sector_2022'] = pd.read_csv("data source/International Energy Agency - CO2 emissions by sector, India, 2022.csv")
            data['sector_historical'] = pd.read_csv("data source/International Energy Agency - CO2 emissions by sector in India.csv")
            data['per_capita'] = pd.read_csv("data source/International Energy Agency - CO2 emissions per capita, Asia Pacific.csv")
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def train_model(self, df):
        """Train the Random Forest model"""
        features = ['temperature', 'travel_km', 'electricity_units', 'food_impact',
                   'waste_kg', 'construction_score', 'agriculture_score', 'lifestyle_score']
        target = 'co2_emission_kg'
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.feature_names = X.columns
        self.model_metrics = {
            'mse': mse,
            'r2': r2,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return self.model, mse, r2
    
    def predict_footprint(self, **kwargs):
        """Predict carbon footprint based on input parameters"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        input_data = pd.DataFrame([list(kwargs.values())], columns=self.feature_names)
        return self.model.predict(input_data)[0]
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            return None
        
        importances = self.model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return feature_df

class ChartGenerator:
    """Class for generating various charts and visualizations"""
    
    @staticmethod
    def create_fuel_pie_chart(fuel_df):
        """Create pie chart for fuel emissions"""
        fuel_summary = fuel_df.groupby("CO2 emissions by fuel, India, 2022")['Value'].sum().reset_index()
        fuel_summary['percentage'] = 100 * fuel_summary['Value'] / fuel_summary['Value'].sum()
        
        fig = px.pie(fuel_summary, 
                     values='percentage', 
                     names='CO2 emissions by fuel, India, 2022',
                     title='CO₂ Emissions by Fuel Type (2022)',
                     color_discrete_sequence=['#FF6F3C', '#FFBD3C', '#3CC9F0', '#8AC926'])
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500, font_size=12)
        
        return fig
    
    @staticmethod
    def create_sector_horizontal_bar(sector_df):
        """Create horizontal bar chart for sector emissions"""
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
                     color='Percentage',
                     color_continuous_scale='Viridis')
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    @staticmethod
    def create_historical_trend(fuel_historical):
        """Create historical trend line chart"""
        fuel_historical.columns = ['Fuel', 'Value', 'Year', 'Units']
        fuel_historical['Year'] = pd.to_numeric(fuel_historical['Year'], errors='coerce')
        fuel_historical = fuel_historical.dropna()
        
        fig = px.line(fuel_historical, 
                      x='Year', 
                      y='Value', 
                      color='Fuel',
                      title='Evolution of CO₂ Emissions by Fuel (2000-2022)',
                      labels={'Value': 'CO₂ Emissions (Mt)'})
        
        fig.update_layout(height=500, hovermode='x unified')
        fig.update_traces(line=dict(width=3), marker=dict(size=6))
        
        return fig
    
    @staticmethod
    def create_feature_importance_chart(feature_df):
        """Create feature importance chart"""
        fig = px.bar(feature_df, 
                     x='Importance', 
                     y='Feature', 
                     orientation='h',
                     title='Feature Importance in CO₂ Prediction',
                     color='Importance',
                     color_continuous_scale='RdYlBu_r')
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    @staticmethod
    def create_predictions_scatter(y_test, y_pred):
        """Create scatter plot for predictions vs actual"""
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={'x': 'Actual CO₂ Emissions (kg)', 'y': 'Predicted CO₂ Emissions (kg)'},
                        title='Model Predictions vs Actual Values')
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash", width=2)
        )
        
        fig.update_layout(height=500)
        return fig

class CarbonTips:
    """Class for providing carbon reduction tips and insights"""
    
    TIPS = {
        'high_travel': [
            "🚲 Consider using public transport, walking, or cycling for short trips",
            "🚗 Combine multiple errands into one trip",
            "🏠 Work from home when possible to reduce commuting",
            "✈️ Choose direct flights and consider carbon offsets for air travel"
        ],
        'high_electricity': [
            "💡 Switch to LED bulbs and energy-efficient appliances",
            "🌡️ Use programmable thermostats and adjust temperature settings",
            "🔌 Unplug devices when not in use",
            "☀️ Consider solar panels or renewable energy options"
        ],
        'high_food': [
            "🌱 Eat more plant-based meals and reduce meat consumption",
            "🥬 Choose locally sourced and seasonal foods",
            "🗑️ Reduce food waste by planning meals and proper storage",
            "🍽️ Consider smaller portion sizes"
        ],
        'high_waste': [
            "♻️ Recycle and compost organic waste",
            "🛍️ Use reusable bags, bottles, and containers",
            "📦 Buy products with minimal packaging",
            "🔄 Donate or sell items instead of throwing them away"
        ],
        'general': [
            "🌳 Plant trees or support reforestation projects",
            "💰 Invest in carbon offset programs",
            "🏠 Improve home insulation and weatherproofing",
            "📱 Use digital receipts and bills to reduce paper use",
            "👥 Educate others about climate change and carbon reduction"
        ]
    }
    
    @classmethod
    def get_personalized_tips(cls, footprint_data):
        """Get personalized tips based on user's footprint data"""
        tips = []
        
        # Analyze high-impact areas
        if footprint_data.get('travel_km', 0) > 100:
            tips.extend(cls.TIPS['high_travel'])
        
        if footprint_data.get('electricity_units', 0) > 400:
            tips.extend(cls.TIPS['high_electricity'])
        
        if footprint_data.get('food_impact', 0) > 6:
            tips.extend(cls.TIPS['high_food'])
        
        if footprint_data.get('waste_kg', 0) > 10:
            tips.extend(cls.TIPS['high_waste'])
        
        # Always include general tips
        tips.extend(cls.TIPS['general'][:3])
        
        return list(set(tips))  # Remove duplicates
    
    @classmethod
    def get_impact_level(cls, footprint, average_footprint):
        """Determine impact level and appropriate message"""
        if footprint < average_footprint * 0.8:
            return "low", "🌟 Excellent! Your carbon footprint is below average."
        elif footprint < average_footprint * 1.2:
            return "medium", "📊 Your carbon footprint is around average."
        elif footprint < average_footprint * 1.5:
            return "high", "⚠️ Your carbon footprint is above average. Consider reducing your impact."
        else:
            return "very_high", "🚨 Your carbon footprint is significantly above average. Immediate action recommended!"

def format_number(num):
    """Format numbers for better display"""
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return f"{num:.1f}"
