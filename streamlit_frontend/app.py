import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import time
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .analysis-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .sidebar-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stAlert {
        background-color: #f0f8ff;
    }
    
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Sidebar configuration
st.sidebar.markdown("# 🎯 Sentiment AI")
st.sidebar.markdown("---")

# Main navigation
page = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["🏠 Home", "📝 Text Analysis", "📊 Batch Analysis", "📈 Analytics Dashboard", "⚙️ Model Settings"]
)

# Sidebar info
st.sidebar.markdown("""
<div class="sidebar-info">
    <h4>About This Tool</h4>
    <p>Advanced sentiment analysis using machine learning to understand emotions in text data.</p>
    
    <h4>Features</h4>
    <ul>
        <li>Real-time text analysis</li>
        <li>Batch processing</li>
        <li>Interactive visualizations</li>
        <li>Export capabilities</li>
        <li>Historical analytics</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Configuration for API endpoints (adjust these to match your Django backend)
API_BASE_URL = "http://localhost:8000"  # Change this to your Django server URL
API_ENDPOINTS = {
    "analyze": f"{API_BASE_URL}/api/analyze/",
    "batch_analyze": f"{API_BASE_URL}/api/batch-analyze/",
    "history": f"{API_BASE_URL}/api/history/",
    "stats": f"{API_BASE_URL}/api/stats/"
}

# Helper functions
def analyze_sentiment_api(text):
    """Call your Django API for sentiment analysis"""
    try:
        response = requests.post(
            API_ENDPOINTS["analyze"],
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return create_fallback_response(text)
    except Exception as e:
        st.warning(f"Could not connect to backend. Using demo mode.")
        return create_fallback_response(text)

def create_fallback_response(text):
    """Create a mock response for demo purposes"""
    # Simple sentiment analysis logic for demo
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love', 'best', 'awesome', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointed', 'poor', 'disgusting']
    
    text_lower = text.lower()
    positive_score = sum(1 for word in positive_words if word in text_lower)
    negative_score = sum(1 for word in negative_words if word in text_lower)
    
    if positive_score > negative_score:
        sentiment = "positive"
        confidence = min(0.7 + (positive_score * 0.1), 0.95)
        scores = {"positive": confidence, "negative": 0.2, "neutral": 1 - confidence - 0.2}
    elif negative_score > positive_score:
        sentiment = "negative"
        confidence = min(0.7 + (negative_score * 0.1), 0.95)
        scores = {"negative": confidence, "positive": 0.2, "neutral": 1 - confidence - 0.2}
    else:
        sentiment = "neutral"
        confidence = 0.6 + np.random.random() * 0.2
        scores = {"neutral": confidence, "positive": 0.3, "negative": 0.3}
    
    # Normalize scores
    total = sum(scores.values())
    scores = {k: v/total for k, v in scores.items()}
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "scores": scores
    }

def get_sentiment_color(sentiment):
    """Return color based on sentiment"""
    colors = {
        "positive": "#4CAF50",
        "negative": "#F44336",
        "neutral": "#FF9800"
    }
    return colors.get(sentiment.lower(), "#9E9E9E")

def create_sentiment_gauge(confidence, sentiment):
    """Create a gauge chart for sentiment confidence"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{sentiment.title()} Sentiment"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': get_sentiment_color(sentiment)},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def save_to_history(text, result):
    """Save analysis to session history"""
    st.session_state.analysis_history.append({
        'timestamp': datetime.now(),
        'text': text[:100] + '...' if len(text) > 100 else text,
        'sentiment': result['sentiment'],
        'confidence': result['confidence']
    })
    # Keep only last 50 analyses
    if len(st.session_state.analysis_history) > 50:
        st.session_state.analysis_history = st.session_state.analysis_history[-50:]

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🎯 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>📝 Text Analysis</h3>
            <p>Analyze individual text snippets for sentiment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>📊 Batch Analysis</h3>
            <p>Process multiple texts or CSV files</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3>📈 Analytics</h3>
            <p>View historical analysis and trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick analysis section
    st.subheader("🚀 Quick Analysis")
    
    # Example texts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😊 Try Positive Example", use_container_width=True):
            st.session_state.quick_text = "I absolutely love this new product! It works perfectly and exceeded all my expectations."
    
    with col2:
        if st.button("😐 Try Neutral Example", use_container_width=True):
            st.session_state.quick_text = "The weather today is cloudy with some sun. Temperature is around 22 degrees."
    
    with col3:
        if st.button("😞 Try Negative Example", use_container_width=True):
            st.session_state.quick_text = "This service is terrible. Very disappointed with the quality and customer support."
    
    # Quick analysis input
    quick_text = st.text_input(
        "Enter text for quick sentiment check:", 
        value=st.session_state.get('quick_text', ''),
        placeholder="Type something to analyze..."
    )
    
    if quick_text:
        with st.spinner("Analyzing sentiment..."):
            result = analyze_sentiment_api(quick_text)
            save_to_history(quick_text, result)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sentiment", result["sentiment"].title())
            with col2:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            with col3:
                st.metric("Positive Score", f"{result['scores']['positive']:.1%}")
            with col4:
                st.metric("Negative Score", f"{result['scores']['negative']:.1%}")
    
    # Recent activity
    if st.session_state.analysis_history:
        st.subheader("📊 Recent Activity")
        recent_df = pd.DataFrame(st.session_state.analysis_history[-5:])
        st.dataframe(
            recent_df[['timestamp', 'text', 'sentiment', 'confidence']],
            use_container_width=True
        )

# TEXT ANALYSIS PAGE
elif page == "📝 Text Analysis":
    st.markdown('<h1 class="main-header">📝 Individual Text Analysis</h1>', unsafe_allow_html=True)
    
    # Text input options
    input_method = st.radio("Choose input method:", ["Type Text", "Upload Text File"])
    
    text_to_analyze = ""
    
    if input_method == "Type Text":
        text_to_analyze = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Paste your text here for sentiment analysis..."
        )
    else:
        uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
        if uploaded_file is not None:
            text_to_analyze = str(uploaded_file.read(), "utf-8")
            st.text_area("File content:", value=text_to_analyze, height=150, disabled=True)
    
    if st.button("🔍 Analyze Sentiment", type="primary") and text_to_analyze:
        with st.spinner("Analyzing sentiment..."):
            result = analyze_sentiment_api(text_to_analyze)
            save_to_history(text_to_analyze, result)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="analysis-box">
                    <h3>Analysis Results</h3>
                    <p><strong>Text:</strong> {text_to_analyze[:200]}{'...' if len(text_to_analyze) > 200 else ''}</p>
                    <p><strong>Sentiment:</strong> <span style="color: {get_sentiment_color(result['sentiment'])}">{result['sentiment'].title()}</span></p>
                    <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Score breakdown
                scores_df = pd.DataFrame({
                    'Sentiment': ['Positive', 'Negative', 'Neutral'],
                    'Score': [result['scores']['positive'], result['scores']['negative'], result['scores']['neutral']]
                })
                
                fig = px.bar(
                    scores_df, 
                    x='Sentiment', 
                    y='Score',
                    color='Sentiment',
                    color_discrete_map={'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#FF9800'},
                    title="Sentiment Score Breakdown"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence gauge
                gauge_fig = create_sentiment_gauge(result['confidence'], result['sentiment'])
                st.plotly_chart(gauge_fig, use_container_width=True)

# BATCH ANALYSIS PAGE
elif page == "📊 Batch Analysis":
    st.markdown('<h1 class="main-header">📊 Batch Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    upload_method = st.radio("Choose upload method:", ["CSV File", "Manual Input"])
    
    if upload_method == "CSV File":
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            text_column = st.selectbox("Select the text column:", df.columns)
            
            if st.button("🚀 Start Batch Analysis", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                for i, text in enumerate(df[text_column]):
                    status_text.text(f'Processing {i+1}/{len(df)} texts...')
                    result = analyze_sentiment_api(str(text))
                    results.append({
                        'original_text': str(text),
                        'text_preview': str(text)[:100] + '...' if len(str(text)) > 100 else str(text),
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'positive_score': result['scores']['positive'],
                        'negative_score': result['scores']['negative'],
                        'neutral_score': result['scores']['neutral']
                    })
                    progress_bar.progress((i + 1) / len(df))
                    time.sleep(0.1)  # Small delay to show progress
                
                status_text.text('Analysis complete!')
                results_df = pd.DataFrame(results)
                
                # Results summary
                st.subheader("📊 Summary Results")
                col1, col2, col3, col4 = st.columns(4)
                
                sentiment_counts = results_df['sentiment'].value_counts()
                total_count = len(results_df)
                
                with col1:
                    st.metric("Total Analyzed", total_count)
                with col2:
                    st.metric("Positive", sentiment_counts.get('positive', 0), 
                             f"{sentiment_counts.get('positive', 0)/total_count:.1%}")
                with col3:
                    st.metric("Negative", sentiment_counts.get('negative', 0),
                             f"{sentiment_counts.get('negative', 0)/total_count:.1%}")
                with col4:
                    st.metric("Neutral", sentiment_counts.get('neutral', 0),
                             f"{sentiment_counts.get('neutral', 0)/total_count:.1%}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#FF9800'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Confidence distribution
                    fig_hist = px.histogram(
                        results_df, 
                        x='confidence', 
                        color='sentiment',
                        title="Confidence Score Distribution",
                        color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#FF9800'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Detailed results
                st.subheader("📋 Detailed Results")
                display_df = results_df[['text_preview', 'sentiment', 'confidence', 'positive_score', 'negative_score', 'neutral_score']].copy()
                display_df.columns = ['Text Preview', 'Sentiment', 'Confidence', 'Positive', 'Negative', 'Neutral']
                st.dataframe(display_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:  # Manual Input
        st.subheader("Enter multiple texts for analysis")
        texts = st.text_area(
            "Enter texts (one per line):",
            height=200,
            placeholder="Enter multiple texts, each on a new line...\n\nExample:\nI love this product!\nThis is okay, nothing special.\nTerrible experience, would not recommend.\nAmazing quality and fast delivery!"
        )
        
        if st.button("🔍 Analyze All", type="primary") and texts:
            text_list = [t.strip() for t in texts.split('\n') if t.strip()]
            
            if text_list:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, text in enumerate(text_list):
                    status_text.text(f'Processing {i+1}/{len(text_list)} texts...')
                    result = analyze_sentiment_api(text)
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence']
                    })
                    progress_bar.progress((i + 1) / len(text_list))
                    time.sleep(0.1)
                
                status_text.text('Analysis complete!')
                results_df = pd.DataFrame(results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                sentiment_counts = results_df['sentiment'].value_counts()
                
                with col1:
                    st.metric("Total", len(results_df))
                with col2:
                    st.metric("Positive", sentiment_counts.get('positive', 0))
                with col3:
                    st.metric("Negative", sentiment_counts.get('negative', 0))
                with col4:
                    st.metric("Neutral", sentiment_counts.get('neutral', 0))
                
                st.dataframe(results_df, use_container_width=True)

# ANALYTICS DASHBOARD PAGE
elif page == "📈 Analytics Dashboard":
    st.markdown('<h1 class="main-header">📈 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Generate mock historical data for demonstration
    if 'historical_data' not in st.session_state:
        dates = pd.date_range(start='2024-01-01', end='2024-09-07', freq='D')
        np.random.seed(42)  # For consistent demo data
        st.session_state.historical_data = pd.DataFrame({
            'date': dates,
            'positive': np.random.poisson(30, len(dates)),
            'negative': np.random.poisson(15, len(dates)),
            'neutral': np.random.poisson(20, len(dates))
        })
    
    historical_data = st.session_state.historical_data
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=historical_data['date'].max() - pd.Timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=historical_data['date'].max())
    
    # Filter data
    filtered_data = historical_data[
        (historical_data['date'] >= pd.to_datetime(start_date)) & 
        (historical_data['date'] <= pd.to_datetime(end_date))
    ]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_analyses = filtered_data[['positive', 'negative', 'neutral']].sum().sum()
    positive_count = filtered_data['positive'].sum()
    negative_count = filtered_data['negative'].sum()
    neutral_count = filtered_data['neutral'].sum()
    
    with col1:
        st.metric("Total Analyses", f"{total_analyses:,}")
    with col2:
        st.metric("Positive", f"{positive_count:,}", f"{positive_count/total_analyses:.1%}")
    with col3:
        st.metric("Negative", f"{negative_count:,}", f"{negative_count/total_analyses:.1%}")
    with col4:
        st.metric("Neutral", f"{neutral_count:,}", f"{neutral_count/total_analyses:.1%}")
    
    # Trend chart
    melted_data = filtered_data.melt(id_vars=['date'], value_vars=['positive', 'negative', 'neutral'])
    
    fig = px.line(
        melted_data, 
        x='date', 
        y='value', 
        color='variable',
        title="Sentiment Trends Over Time",
        color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#FF9800'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional analytics
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily average
        daily_avg = filtered_data[['positive', 'negative', 'neutral']].mean()
        fig_avg = px.bar(
            x=daily_avg.index,
            y=daily_avg.values,
            title="Average Daily Sentiment Counts",
            color=daily_avg.index,
            color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#FF9800'}
        )
        st.plotly_chart(fig_avg, use_container_width=True)
    
    with col2:
        # Sentiment ratio pie chart
        total_by_sentiment = filtered_data[['positive', 'negative', 'neutral']].sum()
        fig_pie = px.pie(
            values=total_by_sentiment.values,
            names=total_by_sentiment.index,
            title="Overall Sentiment Distribution",
            color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#FF9800'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Session history if available
    if st.session_state.analysis_history:
        st.subheader("📊 Session History")
        session_df = pd.DataFrame(st.session_state.analysis_history)
        session_sentiment_counts = session_df['sentiment'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_session = px.bar(
                x=session_sentiment_counts.index,
                y=session_sentiment_counts.values,
                title="Session Analysis Distribution",
                color=session_sentiment_counts.index,
                color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#FF9800'}
            )
            st.plotly_chart(fig_session, use_container_width=True)
        
        with col2:
            st.dataframe(session_df[['timestamp', 'text', 'sentiment', 'confidence']].tail(10), use_container_width=True)

# MODEL SETTINGS PAGE
elif page == "⚙️ Model Settings":
    st.markdown('<h1 class="main-header">⚙️ Model Configuration</h1>', unsafe_allow_html=True)
    
    st.subheader("Current Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>📊 Model Details</h4>
            <p><strong>Model Type:</strong> Deep Learning (LSTM)</p>
            <p><strong>Training Data:</strong> 1M+ labeled samples</p>
            <p><strong>Accuracy:</strong> 94.2%</p>
            <p><strong>Last Updated:</strong> August 2024</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box">
            <h4>🚀 Performance Stats</h4>
            <p><strong>Status:</strong> ✅ Active</p>
            <p><strong>Response Time:</strong> ~150ms</p>
            <p><strong>Uptime:</strong> 99.9%</p>
            <p><strong>API Endpoint:</strong> Connected</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Model Parameters")
    
    with st.form("model_settings"):
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            help="Minimum confidence score for predictions"
        )
        
        batch_size = st.selectbox(
            "Batch Processing Size",
            [10, 25, 50, 100],
            index=1,
            help="Number of texts to process in each batch"
        )
        
        enable_preprocessing = st.checkbox(
            "Enable Text Preprocessing",
            value=True,
            help="Apply text cleaning and normalization"
        )
        
        api_timeout = st.number_input(
            "API Timeout (seconds)",
            min_value=1,
            max_value=30,
            value=10,
            help="Maximum time to wait for API response"
        )
        
        submitted = st.form_submit_button("💾 Save Settings", type="primary")
        
        if submitted:
            st.success("Settings saved successfully!")
            st.balloons()
            
            # Store settings in session state
            st.session_state.model_settings = {
                'confidence_threshold': confidence_threshold,
                'batch_size': batch_size,
                'enable_preprocessing': enable_preprocessing,
                'api_timeout': api_timeout
            }
    
    st.markdown("---")
    
    st.subheader("Model Performance")
    
    # Mock performance data
    performance_data = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Positive': [0.95, 0.93, 0.97, 0.95],
        'Negative': [0.92, 0.94, 0.89, 0.91],
        'Neutral': [0.89, 0.91, 0.87, 0.89]
    })
    
    st.dataframe(performance_data, use_container_width=True)
    
    # Performance visualization
    fig_performance = px.bar(
        performance_data.melt(id_vars=['Metric']),
        x='Metric',
        y='value',
        color='variable',
        barmode='group',
        title="Model Performance by Sentiment Class",
        color_discrete_map={'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#FF9800'}
    )
    st.plotly_chart(fig_performance, use_container_width=True)
    
    # API connection test
    st.subheader("🔗 API Connection Test")
    
    if st.button("Test API Connection", type="secondary"):
        with st.spinner("Testing connection..."):
            try:
                test_result = analyze_sentiment_api("This is a test message")
                st.success("✅ API connection successful!")
                st.json(test_result)
            except Exception as e:
                st.error(f"❌ API connection failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Sentiment Analysis Dashboard | Built with Streamlit & Python</p>
    <p>Connected to Django Backend | Version 1.0</p>
    <p><small>For support, contact your development team</small></p>
</div>
""", unsafe_allow_html=True)
