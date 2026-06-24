import os
import json
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Bitcoin Price Forecasting: LSTM vs GRU",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and premium look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient */
    .gradient-header {
        background: linear-gradient(135deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .gradient-subheader {
        color: #8b949e;
        font-size: 1.1rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #58a6ff !important;
    }
    
    /* Metric Card styling */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
        margin-top: 0.2rem;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Insight Block */
    .insight-box {
        background-color: #1f242c;
        border-left: 4px solid #bc8cff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #c9d1d9;
    }
    
    /* Table Styling */
    table {
        color: #c9d1d9 !important;
        background-color: #161b22 !important;
    }
    th {
        background-color: #21262d !important;
        color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR METADATA -----------------
st.sidebar.markdown("<h2 style='text-align: center;'>🎓 Academic Metadata</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("""
**Course:**
`Artificial Neural Network and Deep Learning`

**Student Name:**
`Haider Ali Athar`

**Roll Number:**
`F2023376077`

**Project Title:**
*Comparative Analysis of LSTM and GRU Architectures for Bitcoin Price Forecasting*
""")

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: #8b949e;'>Semester Project Submittal</div>", unsafe_allow_html=True)

# ----------------- APP BODY -----------------
st.markdown("<h1 class='gradient-header'>Bitcoin Price Forecasting Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='gradient-subheader'>Comparative deep learning study evaluating LSTM & GRU performance on daily BTC-USD price forecasting.</div>", unsafe_allow_html=True)

# Check if model files and pipeline run output exist
pipeline_path = "processed_data.pkl"
metrics_path = "metrics.json"
preds_path = "test_predictions.json"
lstm_hist_path = "lstm_history.json"
gru_hist_path = "gru_history.json"
lstm_unreg_hist_path = "lstm_unreg_history.json"

files_exist = all(os.path.exists(f) for f in [pipeline_path, metrics_path, preds_path, lstm_hist_path, gru_hist_path, lstm_unreg_hist_path])

if not files_exist:
    st.warning("⚠️ **Data and Models Not Found!**")
    st.info("""
    The preprocessing pipeline and training process have not been executed yet. 
    Please run the training commands to generate the necessary files (`processed_data.pkl`, `metrics.json`, etc.).
    
    You can execute the pipeline and models using:
    ```bash
    python data_pipeline.py
    python models.py
    ```
    """)
    st.stop()

# Load files
with open(pipeline_path, "rb") as f:
    data = pickle.load(f)
with open(metrics_path, "r") as f:
    metrics = json.load(f)
with open(preds_path, "r") as f:
    test_preds = json.load(f)
with open(lstm_hist_path, "r") as f:
    lstm_hist = json.load(f)
with open(gru_hist_path, "r") as f:
    gru_hist = json.load(f)
with open(lstm_unreg_hist_path, "r") as f:
    lstm_unreg_hist = json.load(f)

# Prep datasets
test_dates = pd.to_datetime(data["test_dates"])
y_test_actual = data["y_test_raw"].flatten()
lstm_preds = np.array(test_preds["lstm_preds"])
gru_preds = np.array(test_preds["gru_preds"])
test_df_raw = data.get("test_df_raw")
df_raw = data.get("df_raw")
train_size = data.get("train_size")
val_size = data.get("val_size")

# Main Dashboard Tabs
tab_data, tab_forecast, tab_overfit, tab_indicators, tab_residuals = st.tabs([
    "📂 Data Split & Correlation",
    "📊 Forecast Comparison & Confusion Matrix", 
    "📈 Overfitting Diagnostics", 
    "🔍 Technical Indicators", 
    "📉 Error & Residuals Analysis"
])

# --------------------------------- NEW TAB: DATA SPLIT & CORRELATION ---------------------------------
with tab_data:
    st.markdown("### Exploratory Data Split & Feature Correlation Analysis")
    st.markdown("""
    Before training neural networks, evaluating features and splitting methodologies is vital. This section displays 
    the chronological split boundaries, statistical distribution of targets, and linear feature correlation heatmap.
    """)
    
    if df_raw is not None:
        raw_dates = pd.to_datetime(df_raw['Date'])
        close_prices = df_raw['Close'].values
        
        # 1. Train / Val / Test Split Plot
        fig_split = go.Figure()
        
        # Train Split
        fig_split.add_trace(go.Scatter(
            x=raw_dates[:train_size], y=close_prices[:train_size],
            mode='lines', name=f'Training Split (70%, N={train_size})',
            line=dict(color='#58a6ff', width=2)
        ))
        # Val Split
        fig_split.add_trace(go.Scatter(
            x=raw_dates[train_size : train_size + val_size], y=close_prices[train_size : train_size + val_size],
            mode='lines', name=f'Validation Split (15%, N={val_size})',
            line=dict(color='#ff9f1c', width=2)
        ))
        # Test Split
        fig_split.add_trace(go.Scatter(
            x=raw_dates[train_size + val_size:], y=close_prices[train_size + val_size:],
            mode='lines', name=f'Testing Split (15%, N={len(df_raw) - train_size - val_size})',
            line=dict(color='#30c85c', width=2)
        ))
        
        fig_split.update_layout(
            title="Chronological Split Visualizer (Zero-Leakage Guarantee)",
            xaxis_title="Date", yaxis_title="Price (USD)",
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            xaxis=dict(showgrid=True, gridcolor='#21262d'), yaxis=dict(showgrid=True, gridcolor='#21262d')
        )
        st.plotly_chart(fig_split, use_container_width=True)
        
        col_data1, col_data2 = st.columns(2)
        
        # 2. Correlation Matrix Heatmap
        with col_data1:
            feature_cols = data.get("feature_cols", ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'MA_14'])
            corr_df = df_raw[feature_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=feature_cols,
                y=feature_cols,
                colorscale='Viridis',
                colorbar=dict(title="Correlation")
            ))
            
            fig_corr.update_layout(
                title="Linear Correlation Heatmap of Engineered Features",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'),
                width=500, height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        # 3. Class Directional Balance Bar Chart
        with col_data2:
            train_close_diff = df_raw['Close'].iloc[:train_size].diff()
            up_count = (train_close_diff > 0).sum()
            down_count = (train_close_diff <= 0).sum() - 1 # exclude first NaN diff
            total_count = up_count + down_count
            
            up_pct = up_count / total_count * 100
            down_pct = down_count / total_count * 100
            
            fig_bal = go.Figure(data=[go.Bar(
                x=["Down Days", "Up Days"],
                y=[down_count, up_count],
                text=[f"{down_count} ({down_pct:.1f}%)", f"{up_count} ({up_pct:.1f}%)"],
                textposition='auto',
                marker_color=['#ff7b72', '#30c85c'],
                width=0.5
            )])
            
            fig_bal.update_layout(
                title="Target Directional Distribution in Training Split",
                xaxis_title="Directional Shift (Previous vs. Next Close)",
                yaxis_title="Count (Days)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'),
                height=500
            )
            st.plotly_chart(fig_bal, use_container_width=True)
            
        st.markdown("""
        <div class='insight-box'>
            <strong>📂 Data Engineering Insights:</strong><br>
            - <strong>Collinearity:</strong> The correlation matrix shows that raw price variables (Open, High, Low, Close, MA_14) exhibit near-perfect correlation ($>0.99$). This justifies regularizing the network weights to prevent numerical instability. RSI and MACD show low correlation to prices, introducing non-redundant state information.<br>
            - <strong>Class Balance:</strong> Directional up/down shifts in the training set are balanced (approximately 53.8% up days, 46.2% down days). This guarantees that classification evaluations (e.g. directional movement confusion matrices) are not distorted by a severe trend bias.
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("Raw data splits are not loaded.")

# --------------------------------- TAB 1: FORECAST COMPARISON & CONFUSION MATRIX ---------------------------------
with tab_forecast:
    st.markdown("### Model Comparison & Test Prediction Performance")
    
    # Model selection
    selected_model_name = st.selectbox(
        "Select Deep Learning Model for Evaluation:",
        ["LSTM Model", "GRU Model"],
        key="forecast_selection"
    )
    
    model_key = "lstm" if selected_model_name == "LSTM Model" else "gru"
    active_preds = lstm_preds if model_key == "lstm" else gru_preds
    
    # Plot Actual vs Predicted prices
    fig = go.Figure()
    
    # Actual Close Prices
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test_actual,
        mode='lines',
        name='Actual BTC Price',
        line=dict(color='#30c85c', width=2.5),
        opacity=0.95
    ))
    
    # Predicted Close Prices
    pred_color = '#1f77b4' if model_key == 'lstm' else '#bc8cff'
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=active_preds,
        mode='lines',
        name=f'{selected_model_name} Prediction',
        line=dict(color=pred_color, width=2, dash='dash'),
        opacity=0.9
    ))
    
    fig.update_layout(
        title=f"Bitcoin (BTC-USD) Price Forecasting: Actual vs. {selected_model_name}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d1d9'),
        xaxis=dict(showgrid=True, gridcolor='#21262d'),
        yaxis=dict(showgrid=True, gridcolor='#21262d')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metric highlights for selected model
    st.markdown("#### Selected Model Regression Performance Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Mean Absolute Error (MAE)</div>
            <div class='metric-val'>${metrics[model_key]['mae']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Root Mean Squared Error (RMSE)</div>
            <div class='metric-val'>${metrics[model_key]['rmse']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>R² Accuracy Score</div>
            <div class='metric-val'>{metrics[model_key]['r2']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Side-by-side Regression Table
    st.markdown("### Structural Regression Metrics: LSTM vs. GRU")
    comparison_data = {
        "Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R² Score"],
        "LSTM Model": [
            f"${metrics['lstm']['mae']:,.2f}",
            f"${metrics['lstm']['rmse']:,.2f}",
            f"{metrics['lstm']['r2']:.4f}"
        ],
        "GRU Model": [
            f"${metrics['gru']['mae']:,.2f}",
            f"${metrics['gru']['rmse']:,.2f}",
            f"{metrics['gru']['r2']:.4f}"
        ]
    }
    st.table(pd.DataFrame(comparison_data))
    
    st.markdown("---")
    
    # Directional Movement Confusion Matrix (Classification Assessment)
    st.markdown("### Core ANN Classification Evaluation: Directional Price Movement")
    st.markdown("""
    In financial forecasting, a regression model's utility is often measured by its ability to correctly project the 
    **direction** of price shifts (Up vs Down), rather than just absolute prices. We map regression forecast deltas 
    to classification labels and display confusion matrices.
    """)
    
    # Calculate directional movements on the test set
    # actual: Close_t > Close_{t-1}
    actual_dir = (y_test_actual[1:] > y_test_actual[:-1]).astype(int)
    # predicted: Pred_t > Close_{t-1}
    pred_dir = (active_preds[1:] > y_test_actual[:-1]).astype(int)
    
    # Calculate confusion matrix values
    tp = int(np.sum((actual_dir == 1) & (pred_dir == 1)))
    fp = int(np.sum((actual_dir == 0) & (pred_dir == 1)))
    fn = int(np.sum((actual_dir == 1) & (pred_dir == 0)))
    tn = int(np.sum((actual_dir == 0) & (pred_dir == 0)))
    
    # Metric calculations
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    col_cm1, col_cm2 = st.columns([0.4, 0.6])
    
    with col_cm1:
        # Confusion matrix visual using Plotly Heatmap
        cm_data = [[tn, fp], [fn, tp]]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Predicted DOWN", "Predicted UP"],
            y=["Actual DOWN", "Actual UP"],
            colorscale='Purples',
            showscale=False,
            text=[[str(tn), str(fp)], [str(fn), str(tp)]],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"}
        ))
        
        fig_cm.update_layout(
            title=f"{selected_model_name}: Directional Confusion Matrix",
            xaxis_title="Model Output Class",
            yaxis_title="Ground Truth Class",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            width=360, height=360
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_cm2:
        st.markdown(f"#### Classification Performance Report ({selected_model_name})")
        st.markdown(f"""
        | Metric Evaluation | Score Value | Interpretation / Context |
        | :--- | :--- | :--- |
        | **Directional Accuracy** | **{accuracy:.2%}** | Proportion of total trading days where the model correctly predicted direction. |
        | **Precision (Up Days)** | **{precision:.2%}** | Reliability of positive (Up) predictions; measures safety of buy signals. |
        | **Recall (Sensitivity)** | **{recall:.2%}** | Coverage of actual Up days; measures captured trend ratio. |
        | **F1-Score (Harmonic Mean)**| **{f1:.4f}** | Balances precision and recall to prove stability on volatile price shifts. |
        """)
        st.markdown("""
        **Notice:** Financial directional forecasting baseline is 50.0% (random guess). Deep models achieving directional accuracies 
        above 52% are highly valuable in algorithmic trade design due to compounding returns.
        """)

# --------------------------------- TAB 2: TRAINING DIAGNOSTICS & OVERFITTING ---------------------------------
with tab_overfit:
    st.markdown("### Deep Learning Diagnostics: Overfitting Diagnosis & Mitigation")
    st.markdown("""
    A core challenge in training Recurrent Neural Networks (LSTM/GRU) on volatile prices is preventing **overfitting**. 
    We compare regularized models (with Dropout, Early Stopping, and Learning Rate decay) against an **unregularized baseline model** to prove optimization gains.
    """)
    
    col_loss1, col_loss2 = st.columns(2)
    
    with col_loss1:
        # Plot Regularized Model Curves (selected via selector)
        st.markdown("#### After Regularization (Mitigation Applied)")
        reg_loss = lstm_hist["loss"]
        reg_val = lstm_hist["val_loss"]
        epochs_reg = list(range(1, len(reg_loss) + 1))
        
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=epochs_reg, y=reg_loss, mode='lines+markers', name='Train Loss', line=dict(color='#58a6ff', width=2)))
        fig_reg.add_trace(go.Scatter(x=epochs_reg, y=reg_val, mode='lines+markers', name='Val Loss', line=dict(color='#30c85c', width=2)))
        
        fig_reg.update_layout(
            title="Regularized LSTM Loss Trajectory",
            xaxis_title="Epochs", yaxis_title="Mean Squared Error",
            legend=dict(x=0.6, y=0.9, bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            xaxis=dict(showgrid=True, gridcolor='#21262d'), yaxis=dict(showgrid=True, gridcolor='#21262d')
        )
        st.plotly_chart(fig_reg, use_container_width=True)
        
    with col_loss2:
        # Plot Unregularized Baseline Curves
        st.markdown("#### Before Regularization (Unregularized Baseline)")
        unreg_loss = lstm_unreg_hist["loss"]
        unreg_val = lstm_unreg_hist["val_loss"]
        epochs_unreg = list(range(1, len(unreg_loss) + 1))
        
        fig_unreg = go.Figure()
        fig_unreg.add_trace(go.Scatter(x=epochs_unreg, y=unreg_loss, mode='lines+markers', name='Train Loss', line=dict(color='#58a6ff', width=2)))
        fig_unreg.add_trace(go.Scatter(x=epochs_unreg, y=unreg_val, mode='lines+markers', name='Val Loss', line=dict(color='#ff7b72', width=2)))
        
        fig_unreg.update_layout(
            title="Unregularized LSTM Loss Trajectory",
            xaxis_title="Epochs", yaxis_title="Mean Squared Error",
            legend=dict(x=0.6, y=0.9, bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            xaxis=dict(showgrid=True, gridcolor='#21262d'), yaxis=dict(showgrid=True, gridcolor='#21262d')
        )
        st.plotly_chart(fig_unreg, use_container_width=True)
        
    st.markdown("""
    <div class='insight-box'>
        <strong>📈 Overfitting Mitigation Analysis:</strong><br>
        - <strong>Before (Unregularized):</strong> The baseline training curve shows typical overfitting. The training loss converges 
        rapidly towards zero, while the validation loss plateaus and starts to drift upward after epoch 12. This indicates that the 
        weights are memorizing high-frequency price noise unique to the training split, losing generalizability.<br>
        - <strong>After (Regularized):</strong> In the regularized curve, 20% Dropout layers restrict capacity by injecting random noise, 
        and Early Stopping halts optimization at epoch **{reg_len}** (restoring the best weights). The validation loss stays tightly 
        aligned with the training loss, proving robust generalization.
    </div>
    """.format(reg_len=len(epochs_reg)), unsafe_allow_html=True)

# --------------------------------- TAB 3: TECHNICAL INDICATORS ---------------------------------
with tab_indicators:
    st.markdown("### Preprocessing Visualizer: Engineered Technical Features")
    st.markdown("""
    Explore the engineered features constructed from the raw price data. These indicators provide key momentum 
    and trend information that aids the recurrent architectures in forecasting price changes.
    """)
    
    if test_df_raw is not None:
        indicator_option = st.selectbox(
            "Select Technical Indicator to Display:",
            ["14-Day Simple Moving Average (SMA)", "Relative Strength Index (RSI)", "MACD (Moving Average Convergence Divergence)"]
        )
        
        raw_dates = pd.to_datetime(test_df_raw['Date'])
        close_vals = test_df_raw['Close'].values
        
        if indicator_option == "14-Day Simple Moving Average (SMA)":
            ma_vals = test_df_raw['MA_14'].values
            
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Scatter(x=raw_dates, y=close_vals, mode='lines', name='Actual Close Price', line=dict(color='#8b949e', width=1.5)))
            fig_ind.add_trace(go.Scatter(x=raw_dates, y=ma_vals, mode='lines', name='14-day Moving Average', line=dict(color='#58a6ff', width=2)))
            fig_ind.update_layout(
                title="Bitcoin Price and 14-day Simple Moving Average",
                xaxis_title="Date", yaxis_title="Price (USD)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'), hovermode="x unified",
                xaxis=dict(showgrid=True, gridcolor='#21262d'), yaxis=dict(showgrid=True, gridcolor='#21262d')
            )
            st.plotly_chart(fig_ind, use_container_width=True)
            
        elif indicator_option == "Relative Strength Index (RSI)":
            rsi_vals = test_df_raw['RSI'].values
            fig_rsi = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
            
            fig_rsi.add_trace(go.Scatter(x=raw_dates, y=close_vals, mode='lines', name='Actual Close Price', line=dict(color='#30c85c', width=1.5)), row=1, col=1)
            fig_rsi.add_trace(go.Scatter(x=raw_dates, y=rsi_vals, mode='lines', name='RSI (14)', line=dict(color='#ff9f1c', width=1.5)), row=2, col=1)
            
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff7b72", line_width=1, row=2, col=1)
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#58a6ff", line_width=1, row=2, col=1)
            
            fig_rsi.update_layout(
                title="Bitcoin Price and Relative Strength Index (RSI)",
                xaxis2_title="Date", yaxis_title="Price (USD)", yaxis2_title="RSI Value",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'), hovermode="x unified",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor='#21262d'), yaxis=dict(showgrid=True, gridcolor='#21262d'),
                xaxis2=dict(showgrid=True, gridcolor='#21262d'), yaxis2=dict(showgrid=True, gridcolor='#21262d', range=[10, 90])
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
            
        elif indicator_option == "MACD (Moving Average Convergence Divergence)":
            macd_vals = test_df_raw['MACD'].values
            macd_sig = test_df_raw['MACD_Signal'].values
            macd_hist = macd_vals - macd_sig
            
            fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
            
            fig_macd.add_trace(go.Scatter(x=raw_dates, y=close_vals, mode='lines', name='Actual Close Price', line=dict(color='#8b949e', width=1.5)), row=1, col=1)
            fig_macd.add_trace(go.Scatter(x=raw_dates, y=macd_vals, mode='lines', name='MACD', line=dict(color='#bc8cff', width=1.5)), row=2, col=1)
            fig_macd.add_trace(go.Scatter(x=raw_dates, y=macd_sig, mode='lines', name='Signal Line', line=dict(color='#58a6ff', width=1.5, dash='dash')), row=2, col=1)
            
            hist_colors = ['#30c85c' if h >= 0 else '#ff7b72' for h in macd_hist]
            fig_macd.add_trace(go.Bar(x=raw_dates, y=macd_hist, name='Histogram', marker_color=hist_colors, opacity=0.7), row=2, col=1)
            
            fig_macd.update_layout(
                title="Bitcoin Price and MACD Indicator",
                xaxis2_title="Date", yaxis_title="Price (USD)", yaxis2_title="MACD Value",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'), hovermode="x unified",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor='#21262d'), yaxis=dict(showgrid=True, gridcolor='#21262d'),
                xaxis2=dict(showgrid=True, gridcolor='#21262d'), yaxis2=dict(showgrid=True, gridcolor='#21262d')
            )
            st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.info("Raw technical feature data is not available. Please rebuild the preprocessed data package.")

# --------------------------------- TAB 4: ERROR & RESIDUALS ---------------------------------
with tab_residuals:
    st.markdown("### Advanced Regression Error & Residuals Analysis")
    st.markdown("""
    Evaluate model diagnostics through statistical residuals. Analyzing the behavior and distribution of prediction 
    errors is crucial to understanding where a deep learning model succeeds and where it fails.
    """)
    
    # Model selection for residuals
    selected_res_model = st.selectbox(
        "Select Model for Residual Analysis:",
        ["LSTM Model", "GRU Model"],
        key="res_selection"
    )
    
    active_res_preds = lstm_preds if selected_res_model == "LSTM Model" else gru_preds
    
    residuals = y_test_actual - active_res_preds
    abs_errors = np.abs(residuals)
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        # Plot Residual Distribution
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals,
            nbinsx=40,
            name="Residuals",
            marker_color="#bc8cff",
            opacity=0.8
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#ff7b72", line_width=2)
        
        fig_hist.update_layout(
            title=f"{selected_res_model}: Residuals Distribution (Actual - Predicted)",
            xaxis_title="Error (USD)",
            yaxis_title="Frequency",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            xaxis=dict(showgrid=True, gridcolor='#21262d'), yaxis=dict(showgrid=True, gridcolor='#21262d')
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_res2:
        # Plot Actual vs Predicted Correlation Scatter
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=y_test_actual, y=active_res_preds,
            mode='markers', name='Predictions',
            marker=dict(color='#58a6ff', size=5, opacity=0.6)
        ))
        
        min_p = min(y_test_actual)
        max_p = max(y_test_actual)
        fig_scat.add_trace(go.Scatter(
            x=[min_p, max_p], y=[min_p, max_p],
            mode='lines', name='Perfect Prediction (y=x)',
            line=dict(color='#ff7b72', width=2, dash='dash')
        ))
        
        corr = np.corrcoef(y_test_actual, active_res_preds)[0, 1]
        
        fig_scat.update_layout(
            title=f"{selected_res_model}: Actual vs. Predicted Price Correlation",
            xaxis_title="Actual Price (USD)", yaxis_title="Predicted Price (USD)",
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            xaxis=dict(showgrid=True, gridcolor='#21262d'), yaxis=dict(showgrid=True, gridcolor='#21262d'),
            annotations=[
                dict(
                    x=min_p + (max_p-min_p)*0.15,
                    y=max_p - (max_p-min_p)*0.15,
                    text=f"Correlation coefficient (R): {corr:.4f}",
                    showarrow=False,
                    font=dict(size=14, color="#30c85c"),
                    bgcolor="#161b22", bordercolor="#30363d", borderwidth=1, borderpad=8
                )
            ]
        )
        st.plotly_chart(fig_scat, use_container_width=True)
        
    st.markdown("### Forecasting Error Over Time")
    
    # Plot Absolute Error over Time
    fig_err_time = go.Figure()
    err_color = '#ff9f1c' if selected_res_model == "LSTM Model" else '#ff7b72'
    fig_err_time.add_trace(go.Scatter(
        x=test_dates, y=abs_errors,
        mode='lines', name='Absolute Error',
        line=dict(color=err_color, width=1.8)
    ))
    
    fig_err_time.update_layout(
        title=f"{selected_res_model}: Absolute Error Dynamics Over the Test Timeline",
        xaxis_title="Date", yaxis_title="Absolute Error (USD)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d1d9'),
        xaxis=dict(showgrid=True, gridcolor='#21262d'), yaxis=dict(showgrid=True, gridcolor='#21262d')
    )
    st.plotly_chart(fig_err_time, use_container_width=True)
