import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import warnings
warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="AI, Sleep & Developer Productivity",
    page_icon=":computer:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #000000;
        font-size: 1.1rem;
    }
    .insight-box p {
        color: #000000;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .insight-box h3, .insight-box h4 {
        color: #1f77b4;
        font-size: 1.3rem;
    }
    .insight-box li {
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    file_path = "ai_dev_productivity.csv"
    df = pd.read_csv(file_path)
    
    # Create derived features
    df['productivity_index'] = df['commits'] - df['bugs_reported'] + df['task_success']
    df['ai_usage_bin'] = pd.qcut(df['ai_usage_hours'], 4, labels=["Low", "Moderate", "High", "Very High"])
    df['task_success_label'] = df['task_success'].map({0: 'Failed', 1: 'Succeeded'})
    df['commits_per_bug'] = df['commits'] / (df['bugs_reported'] + 1)  # Add 1 to avoid division by zero
    
    return df

@st.cache_resource
def load_model():
    """Load the trained ML model and feature names"""
    try:
        model = joblib.load("task_success_model.pkl")
        feature_info = joblib.load("feature_names.pkl")
        feature_names = feature_info['features']
        return model, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please ensure task_success_model.pkl and feature_names.pkl are in the directory.")
        return None, None

def predict_task_success(model, feature_names, input_data):
    """Make prediction using the loaded model"""
    if model is None:
        return None, None
    
    # Create input array in correct order
    input_array = np.array([[input_data[feature] for feature in feature_names]])
    
    # Make prediction
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]
    
    return prediction, probability

def create_correlation_heatmap(data, title="Correlation Heatmap"):
    """Create a correlation heatmap"""
    # Select only numeric columns for correlation
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def create_scatter_with_regression(data, x, y, title, xlabel, ylabel, color=None):
    """Create scatter plot with regression line"""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=data, x=x, y=y, scatter_kws={"alpha": 0.5}, 
                line_kws={"color": "red"} if color is None else {"color": color}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig

def main():
    # Load data
    df = load_data()
    
    # Header
    st.markdown('<h1 class="main-header">AI, Sleep & Developer Productivity Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Analysis", [
        "Overview & Key Metrics",
        "Detailed Analysis", 
        "ML Model Testing",
        "AI vs Coffee & Success",
        "AI vs Sleep Patterns",
        "Interactive Filters",
        "Raw Data Explorer"
    ])
    
    if page == "Overview & Key Metrics":
        show_overview(df)
    elif page == "Detailed Analysis":
        show_detailed_analysis(df)
    elif page == "ML Model Testing":
        show_model_testing(df)
    elif page == "AI vs Coffee & Success":
        show_coffee_analysis(df)
    elif page == "AI vs Sleep Patterns":
        show_sleep_analysis(df)
    elif page == "Interactive Filters":
        show_interactive_filters(df)
    elif page == "Raw Data Explorer":
        show_raw_data(df)

def show_overview(df):
    st.header("Dataset Overview")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Developers", len(df))
        st.metric("Success Rate", f"{df['task_success'].mean():.1%}")
    
    with col2:
        st.metric("Avg AI Usage", f"{df['ai_usage_hours'].mean():.1f} hrs/day")
        st.metric("Avg Commits", f"{df['commits'].mean():.1f}")
    
    with col3:
        st.metric("Avg Sleep", f"{df['sleep_hours'].mean():.1f} hrs")
        st.metric("Avg Coffee", f"{df['coffee_intake_mg'].mean():.0f} mg")
    
    with col4:
        st.metric("Avg Bugs", f"{df['bugs_reported'].mean():.1f}")
        st.metric("Avg Cognitive Load", f"{df['cognitive_load'].mean():.1f}")
    
    # Key findings
    st.markdown("""
    <div class="insight-box">
    <h3>Key Findings</h3>
    <ul>
    <li><strong>AI usage</strong> shows a <strong>moderate positive correlation</strong> with commits (r = 0.37) and task success (r = 0.24)</li>
    <li><strong>Coffee intake</strong> correlates strongly with task success (r = 0.70)</li>
    <li><strong>Sleep</strong> and cognitive load show strong negative correlation (-0.73)</li>
    <li>AI appears to boost productivity without significantly increasing bugs</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    fig = create_correlation_heatmap(df, "Correlation Heatmap — AI Developer Productivity")
    st.pyplot(fig)

def show_detailed_analysis(df):
    st.header("Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AI Usage vs Task Success")
        fig = create_scatter_with_regression(
            df, "ai_usage_hours", "task_success",
            "AI Usage vs Task Success",
            "AI Usage (hours/day)", "Task Success (0 or 1)"
        )
        st.pyplot(fig)
        
        st.subheader("AI Usage vs Commits")
        fig, ax = plt.subplots(figsize=(8, 5))
        successful = df[df['task_success'] == 1]
        failed = df[df['task_success'] == 0]
        
        ax.scatter(successful['ai_usage_hours'], successful['commits'], 
                  alpha=0.8, color='green', label='Succeeded', s=60)
        ax.scatter(failed['ai_usage_hours'], failed['commits'], 
                  alpha=0.8, color='red', label='Failed', s=60)
        ax.set_title("AI Usage vs Commits (Colored by Task Success)")
        ax.set_xlabel("AI Usage (hours/day)")
        ax.set_ylabel("Commits")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Productivity Index vs AI Usage")
        fig = create_scatter_with_regression(
            df, "ai_usage_hours", "productivity_index",
            "Overall Productivity Index vs AI Usage",
            "AI Usage Hours", "Productivity Index (Commits - Bugs + Success)"
        )
        st.pyplot(fig)
        
        st.subheader("AI Usage Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["ai_usage_hours"], bins=20, color="purple", alpha=0.7, edgecolor='black')
        
        # Add density curve
        x_range = np.linspace(df["ai_usage_hours"].min(), df["ai_usage_hours"].max(), 100)
        y_range = stats.gaussian_kde(df["ai_usage_hours"])(x_range)
        ax.plot(x_range, y_range * len(df["ai_usage_hours"]) * (df["ai_usage_hours"].max() - df["ai_usage_hours"].min()) / 20, 
                color='darkviolet', linewidth=2)
        ax.set_title("Distribution of AI Usage (hours/day)")
        ax.set_xlabel("AI Usage (hours/day)")
        ax.set_ylabel("Number of Developers")
        plt.tight_layout()
        st.pyplot(fig)

def show_coffee_analysis(df):
    st.header("Does AI Make Developers Smarter or Just More Caffeinated?")
    
    st.markdown("""
    <div class="insight-box">
    <p>In a world where every developer has at least one caffeine source and one AI tab open, 
    we ask the real question: <strong>does AI actually make us better at coding — or just busier?</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Aggregate data by AI usage bins
    summary = df.groupby("ai_usage_bin", observed=False).agg({
        "coffee_intake_mg": "mean",
        "task_success": "mean"
    }).reset_index()
    
    # Create dual-axis visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Coffee intake bars
    bars = ax1.bar(summary["ai_usage_bin"], summary["coffee_intake_mg"], 
                   color="#8B4513", alpha=0.7, width=0.6)
    ax1.set_ylabel("Average Coffee Intake (mg)", color="#8B4513", fontsize=12)
    ax1.set_xlabel("AI Usage Level", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="#8B4513")
    
    # Task success line (second axis)
    ax2 = ax1.twinx()
    ax2.plot(summary["ai_usage_bin"], summary["task_success"], 
             color="#32CD32", marker="o", linewidth=3, markersize=10)
    ax2.set_ylabel("Task Success Rate", color="#32CD32", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="#32CD32")
    
    plt.title("AI Usage vs Coffee Intake and Task Success", fontsize=14, weight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    <div class="insight-box">
    <h4>Key Insights:</h4>
    <ul>
    <li>As AI usage increases, both task success rates and coffee intake rise together</li>
    <li>Developers with very high AI usage show the highest success rates (≈ 76%) but also the highest caffeine intake (≈ 530 mg/day)</li>
    <li>The relationship suggests: <strong>AI + Caffeine = Peak Developer Mode</strong></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_sleep_analysis(df):
    st.header("Can AI Replace Sleep?")
    
    # Create scatter plot of AI usage vs sleep, colored by task success
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot failed tasks
    failed_data = df[df['task_success'] == 0]
    ax.scatter(failed_data["sleep_hours"], failed_data["ai_usage_hours"], 
               c="#FF6347", alpha=0.7, label="Failed", s=50)
    
    # Plot succeeded tasks  
    success_data = df[df['task_success'] == 1]
    ax.scatter(success_data["sleep_hours"], success_data["ai_usage_hours"], 
               c="#32CD32", alpha=0.7, label="Succeeded", s=50)
    
    ax.set_title("Sleep Hours vs AI Usage (Colored by Task Success)", fontsize=14, weight="bold")
    ax.set_xlabel("Sleep Hours per Day")
    ax.set_ylabel("AI Usage Hours per Day")
    ax.legend(title="Task Success")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Sleep vs cognitive load correlation
    st.subheader("Sleep, Focus, and Cognitive Load")
    lifestyle_cols = ["coffee_intake_mg", "sleep_hours", "distractions", "cognitive_load"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[lifestyle_cols].corr(), annot=True, cmap="RdBu_r", center=0, fmt='.2f', ax=ax)
    ax.set_title("Correlation Heatmap: Coffee, Sleep, and Focus")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    <div class="insight-box">
    <h4>Key Insights:</h4>
    <ul>
    <li><strong>Moderate sleepers (6–8 hours)</strong> using some AI assistance have the highest success rates</li>
    <li>Developers with <strong>low sleep (&lt;5 hours)</strong> rarely perform well, even with heavy AI use</li>
    <li><strong>Sleep and cognitive load</strong> show strong negative correlation (-0.73)</li>
    <li><strong>AI may make developers faster, but not invincible</strong> — sleep still matters!</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_interactive_filters(df):
    st.header("Interactive Analysis")
    
    st.sidebar.subheader("Filters")
    
    # Filters
    ai_usage_range = st.sidebar.slider(
        "AI Usage Hours", 
        float(df['ai_usage_hours'].min()), 
        float(df['ai_usage_hours'].max()), 
        (float(df['ai_usage_hours'].min()), float(df['ai_usage_hours'].max()))
    )
    
    sleep_range = st.sidebar.slider(
        "Sleep Hours", 
        float(df['sleep_hours'].min()), 
        float(df['sleep_hours'].max()), 
        (float(df['sleep_hours'].min()), float(df['sleep_hours'].max()))
    )
    
    task_success_filter = st.sidebar.multiselect(
        "Task Success", 
        ['Failed', 'Succeeded'], 
        default=['Failed', 'Succeeded']
    )
    
    # Apply filters
    filtered_df = df[
        (df['ai_usage_hours'] >= ai_usage_range[0]) & 
        (df['ai_usage_hours'] <= ai_usage_range[1]) &
        (df['sleep_hours'] >= sleep_range[0]) & 
        (df['sleep_hours'] <= sleep_range[1]) &
        (df['task_success_label'].isin(task_success_filter))
    ]
    
    st.subheader(f"Filtered Results ({len(filtered_df)} developers)")
    
    if len(filtered_df) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Success Rate", f"{filtered_df['task_success'].mean():.1%}")
            st.metric("Avg Commits", f"{filtered_df['commits'].mean():.1f}")
        
        with col2:
            st.metric("Avg Bugs", f"{filtered_df['bugs_reported'].mean():.1f}")
            st.metric("Avg Coffee", f"{filtered_df['coffee_intake_mg'].mean():.0f} mg")
        
        with col3:
            st.metric("Commits/Bug Ratio", f"{filtered_df['commits_per_bug'].mean():.1f}")
            st.metric("Avg Cognitive Load", f"{filtered_df['cognitive_load'].mean():.1f}")
        
        # Custom visualization based on filters
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(filtered_df['ai_usage_hours'], filtered_df['commits'], 
                           c=filtered_df['cognitive_load'], cmap='viridis', 
                           alpha=0.7, s=60)
        ax.set_xlabel("AI Usage Hours")
        ax.set_ylabel("Commits")
        ax.set_title("AI Usage vs Commits (Colored by Cognitive Load)")
        plt.colorbar(scatter, label="Cognitive Load")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No data matches the current filters. Please adjust your selection.")

def show_raw_data(df):
    st.header("Raw Data Explorer")
    
    st.subheader("Dataset Info")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes)
    
    st.subheader("Statistical Summary")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        st.success("No missing values found!")
    else:
        st.write(missing_data[missing_data > 0])
    
    st.subheader("Raw Data")
    st.write("**Search and filter the data:**")
    
    # Search functionality
    search_term = st.text_input("Search in data (enter column name or value):")
    
    if search_term:
        # Search across all columns
        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        filtered_data = df[mask]
        st.write(f"Found {len(filtered_data)} rows matching '{search_term}'")
        st.dataframe(filtered_data)
    else:
        st.dataframe(df)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="ai_dev_productivity_filtered.csv",
        mime="text/csv"
    )

def show_model_testing(df):
    st.header("Machine Learning Model Testing")
    
    # Load model
    model, feature_names = load_model()
    
    if model is None:
        st.error("Model could not be loaded. Please ensure the model files are present.")
        return
    
    st.success("Model loaded successfully!")
    
    # Show model info
    with st.expander("Model Information"):
        st.write("**Model Type:** Random Forest Classifier")
        st.write("**Features:**", feature_names)
        st.write("**Target:** Task Success (0 = Failed, 1 = Succeeded)")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
            ax.set_title("Feature Importance")
            plt.tight_layout()
            st.pyplot(fig)
    
    # Tabs for different testing modes
    tab1, tab2, tab3, tab4 = st.tabs(["Custom Prediction", "Predefined Scenarios", "Batch Testing", "Model Analysis"])
    
    with tab1:
        show_custom_prediction(model, feature_names, df)
    
    with tab2:
        show_predefined_scenarios(model, feature_names)
    
    with tab3:
        show_batch_testing(model, feature_names, df)
    
    with tab4:
        show_model_analysis(model, feature_names, df)

def show_custom_prediction(model, feature_names, df):
    st.subheader("Custom Developer Day Prediction")
    st.write("Adjust the parameters below to predict task success for a developer day:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hours_coding = st.slider("Hours Coding", 0.0, 15.0, 6.0, 0.5)
        coffee_intake = st.slider("Coffee Intake (mg)", 0, 800, 400, 25)
        distractions = st.slider("Distractions (count)", 0, 15, 3, 1)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
    
    with col2:
        commits = st.slider("Commits", 0, 20, 5, 1)
        bugs_reported = st.slider("Bugs Reported", 0, 10, 2, 1)
        ai_usage_hours = st.slider("AI Usage Hours", 0.0, 8.0, 2.0, 0.25)
        cognitive_load = st.slider("Cognitive Load (1-10)", 1.0, 10.0, 5.0, 0.5)
    
    # Create input data
    input_data = {
        'hours_coding': hours_coding,
        'coffee_intake_mg': coffee_intake,
        'distractions': distractions,
        'sleep_hours': sleep_hours,
        'commits': commits,
        'bugs_reported': bugs_reported,
        'ai_usage_hours': ai_usage_hours,
        'cognitive_load': cognitive_load
    }
    
    # Make prediction
    prediction, probability = predict_task_success(model, feature_names, input_data)
    
    if prediction is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.success("SUCCESS PREDICTED")
            else:
                st.error("FAILURE PREDICTED")
        
        with col2:
            st.metric("Success Probability", f"{probability[1]:.1%}")
        
        with col3:
            confidence = "High" if max(probability) > 0.8 else "Medium" if max(probability) > 0.6 else "Low"
            st.metric("Confidence", confidence)
        
        # Analysis
        st.subheader("Analysis")
        analysis_text = generate_prediction_analysis(input_data, prediction, probability[1])
        st.write(analysis_text)
        
        # Comparison with dataset
        st.subheader("Comparison with Dataset")
        show_comparison_with_dataset(input_data, df)

def show_predefined_scenarios(model, feature_names):
    st.subheader("Predefined Scenarios")
    st.write("Test the model with realistic developer scenarios:")
    
    scenarios = {
        "Highly Productive Day": {
            'hours_coding': 8.0, 'coffee_intake_mg': 500, 'distractions': 2,
            'sleep_hours': 7.5, 'commits': 10, 'bugs_reported': 1,
            'ai_usage_hours': 2.5, 'cognitive_load': 4.5
        },
        "Struggling Day": {
            'hours_coding': 3.0, 'coffee_intake_mg': 100, 'distractions': 8,
            'sleep_hours': 4.5, 'commits': 2, 'bugs_reported': 5,
            'ai_usage_hours': 0.5, 'cognitive_load': 8.5
        },
        "Average Day": {
            'hours_coding': 6.0, 'coffee_intake_mg': 400, 'distractions': 3,
            'sleep_hours': 6.0, 'commits': 7, 'bugs_reported': 2,
            'ai_usage_hours': 1.5, 'cognitive_load': 5.0
        },
        "AI-Heavy Day": {
            'hours_coding': 9.0, 'coffee_intake_mg': 550, 'distractions': 1,
            'sleep_hours': 8.0, 'commits': 12, 'bugs_reported': 0,
            'ai_usage_hours': 5.0, 'cognitive_load': 3.0
        },
        "Burnout Warning": {
            'hours_coding': 12.0, 'coffee_intake_mg': 600, 'distractions': 4,
            'sleep_hours': 3.0, 'commits': 8, 'bugs_reported': 4,
            'ai_usage_hours': 3.0, 'cognitive_load': 9.5
        }
    }
    
    for scenario_name, scenario_data in scenarios.items():
        with st.expander(scenario_name):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Input Parameters:**")
                for key, value in scenario_data.items():
                    display_name = key.replace('_', ' ').title()
                    if 'mg' in key:
                        st.write(f"• {display_name}: {value} mg")
                    elif 'hours' in key:
                        st.write(f"• {display_name}: {value} hrs")
                    else:
                        st.write(f"• {display_name}: {value}")
            
            with col2:
                prediction, probability = predict_task_success(model, feature_names, scenario_data)
                
                if prediction is not None:
                    if prediction == 1:
                        st.success(f"SUCCESS ({probability[1]:.1%})")
                    else:
                        st.error(f"FAILURE ({probability[0]:.1%})")
                    
                    # Mini analysis
                    st.write("**Quick Analysis:**")
                    analysis = generate_prediction_analysis(scenario_data, prediction, probability[1])
                    st.write(analysis[:200] + "..." if len(analysis) > 200 else analysis)

def show_batch_testing(model, feature_names, df):
    st.subheader("Batch Testing")
    st.write("Test the model on a subset of the original dataset:")
    
    # Sample selection
    sample_size = st.slider("Sample Size", 10, min(100, len(df)), 20)
    random_state = st.number_input("Random Seed", 0, 1000, 42)
    
    if st.button("Generate Random Sample"):
        sample_df = df.sample(n=sample_size, random_state=random_state)
        
        # Make predictions
        predictions = []
        probabilities = []
        
        for _, row in sample_df.iterrows():
            input_data = {feature: row[feature] for feature in feature_names}
            pred, prob = predict_task_success(model, feature_names, input_data)
            predictions.append(pred)
            probabilities.append(prob[1] if prob is not None else 0)
        
        # Add predictions to dataframe
        results_df = sample_df.copy()
        results_df['predicted_success'] = predictions
        results_df['predicted_probability'] = probabilities
        results_df['correct_prediction'] = (results_df['task_success'] == results_df['predicted_success'])
        
        # Display results
        accuracy = results_df['correct_prediction'].mean()
        st.metric("Accuracy on Sample", f"{accuracy:.1%}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prediction Distribution:**")
            pred_counts = pd.Series(predictions).value_counts()
            st.write(f"• Predicted Success: {pred_counts.get(1, 0)}")
            st.write(f"• Predicted Failure: {pred_counts.get(0, 0)}")
        
        with col2:
            st.write("**Actual Distribution:**")
            actual_counts = sample_df['task_success'].value_counts()
            st.write(f"• Actual Success: {actual_counts.get(1, 0)}")
            st.write(f"• Actual Failure: {actual_counts.get(0, 0)}")
        
        # Show detailed results
        st.subheader("Detailed Results")
        display_cols = ['task_success', 'predicted_success', 'predicted_probability', 'correct_prediction'] + feature_names
        st.dataframe(results_df[display_cols])
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(results_df['predicted_probability'], results_df['task_success'], 
                           c=results_df['correct_prediction'], cmap='RdYlGn', alpha=0.7)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Actual Task Success')
        ax.set_title('Prediction vs Reality')
        plt.colorbar(scatter, label='Correct Prediction')
        st.pyplot(fig)

def show_model_analysis(model, feature_names, df):
    st.subheader("Model Analysis")
    
    # Feature correlation with target
    st.write("**Feature Correlation with Task Success:**")
    correlations = []
    for feature in feature_names:
        corr = df[feature].corr(df['task_success'])
        correlations.append((feature, corr))
    
    corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation']).sort_values('Correlation', key=abs, ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=corr_df, x='Correlation', y='Feature', ax=ax)
    ax.set_title('Feature Correlation with Task Success')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature statistics by success/failure
    st.write("**Feature Statistics by Outcome:**")
    success_stats = df[df['task_success'] == 1][feature_names].mean()
    failure_stats = df[df['task_success'] == 0][feature_names].mean()
    
    comparison_df = pd.DataFrame({
        'Success (Mean)': success_stats,
        'Failure (Mean)': failure_stats,
        'Difference': success_stats - failure_stats
    })
    
    st.dataframe(comparison_df.round(2))
    
    # Model performance on full dataset
    if st.button("Test Model on Full Dataset"):
        with st.spinner("Testing model..."):
            all_predictions = []
            all_probabilities = []
            
            for _, row in df.iterrows():
                input_data = {feature: row[feature] for feature in feature_names}
                pred, prob = predict_task_success(model, feature_names, input_data)
                all_predictions.append(pred)
                all_probabilities.append(prob[1] if prob is not None else 0)
            
            accuracy = (np.array(all_predictions) == df['task_success']).mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Full Dataset Accuracy", f"{accuracy:.1%}")
            with col2:
                avg_confidence = np.mean([max(model.predict_proba([[df.iloc[i][feature] for feature in feature_names]])[0]) for i in range(len(df))])
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
            with col3:
                st.metric("Total Samples", len(df))

def generate_prediction_analysis(input_data, prediction, probability):
    """Generate human-readable analysis of the prediction"""
    analysis = []
    
    if prediction == 1:
        analysis.append(f"This looks like a successful day with {probability:.1%} confidence!")
        
        if probability > 0.8:
            analysis.append("The model is highly confident in this prediction.")
        elif probability > 0.6:
            analysis.append("The model shows good confidence in this prediction.")
        else:
            analysis.append("The model shows moderate confidence - there might be some risk factors.")
    else:
        analysis.append(f"This appears to be a challenging day with {1-probability:.1%} confidence of failure.")
    
    # Analyze key factors
    if input_data['sleep_hours'] < 5:
        analysis.append("Low sleep hours are a major risk factor for productivity.")
    elif input_data['sleep_hours'] > 8:
        analysis.append("Good sleep hours support productivity.")
    
    if input_data['distractions'] > 6:
        analysis.append("High distraction count may impact focus and success.")
    elif input_data['distractions'] < 3:
        analysis.append("Low distraction count supports focused work.")
    
    if input_data['ai_usage_hours'] > 3:
        analysis.append("High AI usage suggests good tool integration.")
    elif input_data['ai_usage_hours'] < 1:
        analysis.append("Consider leveraging AI tools more for productivity gains.")
    
    if input_data['cognitive_load'] > 7:
        analysis.append("High cognitive load indicates potential burnout risk.")
    
    return " ".join(analysis)

def show_comparison_with_dataset(input_data, df):
    """Show how input parameters compare to dataset statistics"""
    comparison_data = []
    
    numeric_features = ['hours_coding', 'coffee_intake_mg', 'distractions', 'sleep_hours', 
                       'commits', 'bugs_reported', 'ai_usage_hours', 'cognitive_load']
    
    for feature in numeric_features:
        if feature in input_data:
            value = input_data[feature]
            mean_val = df[feature].mean()
            percentile = (df[feature] < value).mean() * 100
            
            comparison_data.append({
                'Parameter': feature.replace('_', ' ').title(),
                'Your Value': value,
                'Dataset Average': round(mean_val, 2),
                'Your Percentile': f"{percentile:.0f}%"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)

if __name__ == "__main__":
    main()
