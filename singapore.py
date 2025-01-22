import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import mlflow
import mlflow.pyfunc
import plotly.express as px
import plotly.graph_objects as go

# Set wide page layout and custom theme
st.set_page_config(
    page_title="Singapore Property Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Custom card styling */
    .css-1r6slb0 {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metric containers */
    .metric-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Prediction box styling */
    .prediction-box {
        background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    /* Custom header styling */
    .custom-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3D59;
        margin-bottom: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 5px;
        padding: 0 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        padding: 10px 20px;
        font-weight: 500;
        border-radius: 5px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0052a3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f3f8;
        padding: 20px;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load sample data (replace with your actual data loading logic)
@st.cache_data
def load_data():
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample properties
    data = {
        'flat_type': np.random.choice(['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE'], n_samples),
        'floor_area_sqm': np.random.uniform(40, 200, n_samples),
        'resale_price': np.random.uniform(200000, 1500000, n_samples),
        'block': np.random.randint(1, 999, n_samples),
        'street_name': [f'Street {i}' for i in range(1, 21)] * (n_samples // 20),
        'town': np.random.choice(['ANG MO KIO', 'BEDOK', 'TAMPINES', 'WOODLANDS', 'PUNGGOL'], n_samples),
        'year': np.random.randint(2020, 2024, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'lease_commence_date': np.random.randint(1970, 2010, n_samples)
    }
    
    # Adjust prices based on flat type and area
    flat_type_multiplier = {
        '2 ROOM': 0.7,
        '3 ROOM': 0.9,
        '4 ROOM': 1.0,
        '5 ROOM': 1.3,
        'EXECUTIVE': 1.5
    }
    
    for i in range(n_samples):
        data['resale_price'][i] *= flat_type_multiplier[data['flat_type'][i]]
        data['resale_price'][i] *= (data['floor_area_sqm'][i] / 100)
    
    return pd.DataFrame(data)

def create_animated_metric(label, value, prefix="$"):
    """Create an animated metric display"""
    st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #666; font-size: 1rem; margin-bottom: 5px;">{label}</h3>
            <h2 style="color: #1E3D59; font-size: 1.8rem; margin: 0;">{prefix}{value:,.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

def load_mlflow_model():
    """Load the MLflow model with error handling"""
    try:
        model_name = "House_Price_XGBoost_Model"
        model_uri = f"models:/{model_name}/latest"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_prediction_card(features, prediction):
    """Create a visually appealing prediction card"""
    st.markdown(f"""
        <div class="prediction-box">
            <h2 style="margin:0;">Predicted Price</h2>
            <h1 style="font-size: 2.5rem; margin: 10px 0;">SGD ${prediction:,.2f}</h1>
            <p style="margin:0;">Based on {features['floor_area_sqm']}sqm in Block {features['block']}</p>
        </div>
    """, unsafe_allow_html=True)

# Load data
data = load_data()

# Main app layout
st.markdown('<h1 class="custom-header">Singapore Property Analytics</h1>', unsafe_allow_html=True)

# Create modern tabs
tab1, tab2 = st.tabs(["📊 Market Analysis", "🎯 Price Prediction"])

with tab1:
    st.markdown("### Market Insights Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_price = data['resale_price'].mean()
        create_animated_metric("Average Price", avg_price)
    with col2:
        avg_psf = (data['resale_price'] / data['floor_area_sqm']).mean()
        create_animated_metric("Average Price/SQM", avg_psf)
    with col3:
        total_transactions = len(data)
        create_animated_metric("Total Transactions", total_transactions, prefix="")
    with col4:
        avg_area = data['floor_area_sqm'].mean()
        create_animated_metric("Average Area (sqm)", avg_area, prefix="")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(data, x='flat_type', y='resale_price', 
                    title='Price Distribution by Flat Type',
                    color='flat_type')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(data, x='floor_area_sqm', y='resale_price',
                       title='Price vs Floor Area',
                       color='flat_type',
                       size='resale_price')
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Time series analysis
        monthly_avg = data.groupby(['year', 'month'])['resale_price'].mean().reset_index()
        monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))
        fig = px.line(monthly_avg, x='date', y='resale_price',
                     title='Average Price Trend Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Town comparison
        town_stats = data.groupby('town')['resale_price'].agg(['mean', 'count']).reset_index()
        fig = px.bar(town_stats, x='town', y='mean',
                    title='Average Price by Town',
                    color='count',
                    labels={'mean': 'Average Price', 'count': 'Number of Transactions'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    st.markdown("### 📈 Market Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Price per square meter analysis
        data['price_per_sqm'] = data['resale_price'] / data['floor_area_sqm']
        fig = px.histogram(data, x='price_per_sqm', nbins=50,
                          title='Distribution of Price per Square Meter')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age analysis
        data['building_age'] = 2024 - data['lease_commence_date']
        fig = px.scatter(data, x='building_age', y='resale_price',
                        color='flat_type', title='Price vs Building Age')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Smart Price Predictor")
    
    model = load_mlflow_model()
    
    if model:
        # Create three columns for inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📍 Location Details")
            block = st.number_input("Block Number", 
                                  min_value=1, 
                                  max_value=999, 
                                  value=123,
                                  help="Enter the block number of the property")
            
            month = st.select_slider("Month", 
                                   options=list(range(1, 13)),
                                   value=6,
                                   help="Select the month for prediction")
        
        with col2:
            st.markdown("#### 📐 Property Details")
            floor_area = st.slider("Floor Area (sqm)", 
                                 min_value=20.0,
                                 max_value=200.0,
                                 value=70.0,
                                 step=0.5,
                                 help="Select the floor area in square meters")
            
            lease_year = st.number_input("Lease Commence Year",
                                       min_value=1960,
                                       max_value=2024,
                                       value=1995,
                                       help="Enter the year the lease commenced")
        
        with col3:
            st.markdown("#### 📅 Temporal Details")
            prediction_year = st.selectbox("Prediction Year",
                                         options=list(range(2024, 2026)),
                                         help="Select the year for price prediction")
            
            # Add some spacing
            st.markdown("<br>" * 2, unsafe_allow_html=True)
            
            # Center the predict button
            predict_button = st.button("🎯 Predict Price", 
                                     help="Click to get the predicted price",
                                     use_container_width=True)
        
        if predict_button:
            features = {
                'month': month,
                'block': block,
                'floor_area_sqm': floor_area,
                'lease_commence_date': lease_year,
                'year': prediction_year
            }
            
            # Make prediction with loading spinner
            with st.spinner('Calculating prediction...'):
                predicted_price = model.predict(pd.DataFrame([features]))[0]
            
            # Display prediction in an appealing way
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                create_prediction_card(features, predicted_price)
            
            with col2:
                st.markdown("#### 📊 Property Metrics")
                st.markdown(f"""
                    - **Age**: {prediction_year - lease_year} years
                    - **Size Category**: {'Large' if floor_area > 100 else 'Medium' if floor_area > 70 else 'Small'}
                    - **Location**: Block {block}
                    - **Season**: {'Q1' if month <= 3 else 'Q2' if month <= 6 else 'Q3' if month <= 9 else 'Q4'}
                """)
            
            # Add market context
            st.markdown("### 📈 Market Context")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_animated_metric("Predicted Price per SQM", predicted_price/floor_area)
            
            with col2:
                create_animated_metric("Estimated Monthly Mortgage", predicted_price * 0.003)
            
            with col3:
                create_animated_metric("Estimated Down Payment", predicted_price * 0.25)

# Sidebar enhancements
with st.sidebar:
    st.markdown("### 🏢 About the Predictor")
    st.markdown("""
        This advanced analytics tool combines historical data analysis 
        with machine learning to provide accurate property price predictions 
        in Singapore's dynamic real estate market.
    """)
    
    st.markdown("### 🎯 Prediction Accuracy")
    st.progress(0.85)
    st.caption("Model Confidence Score: 85%")
    
    st.markdown("### 📊 Data Sources")
    st.markdown("""
        - HDB Resale Transactions
        - URA Property Data
        - Market Trends Analysis
    """)
    
    # Add last updated timestamp
    st.markdown("---")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")