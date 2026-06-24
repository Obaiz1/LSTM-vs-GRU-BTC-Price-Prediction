# Comparative Analysis of LSTM and GRU Architectures for Bitcoin Price Forecasting

**Course:** Artificial Neural Network and Deep Learning  
**Student Name:** Haider Ali Athar  
**Roll Number:** F2023376077  
**Project Title:** Comparative Analysis of LSTM and GRU Architectures for Bitcoin Price Forecasting  

---

## 1. Abstract & Project Objectives

Predicting cryptocurrency assets, specifically Bitcoin (BTC-USD), represents a significant challenge in financial time-series forecasting due to non-stationarity, high volatility, and multi-scale temporal dependencies. This project evaluates two popular Recurrent Neural Network (RNN) variants: Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks. Using daily historical data spanning more than a decade (2014–2026) obtained via the Yahoo Finance API, we engineer technical features (RSI, MACD, and 14-day MA) and model historical patterns using a 60-day sliding lookback window. The core objectives of this comparative study are:
1. To engineer a leakage-free preprocessing pipeline featuring chronological train/validation/test splits and isolated MinMaxScaler fits.
2. To construct, train, and regularize LSTM and GRU models of comparable parameter complexity.
3. To design and implement overfitting mitigation strategies using Dropout regularization, EarlyStopping, and learning rate scheduling callbacks.
4. To conduct an empirical evaluation comparing MAE, RMSE, and $R^2$ scores to identify which architecture generalizes better on unseen test prices.

---

## 2. Data Preprocessing & Sequence Engineering

### 2.1 Feature Engineering & Technical Indicators
Financial price forecasting benefits from integrating market momentum indicators alongside raw close prices:
- **Relative Strength Index (RSI):** Captures overbought ($>70$) or oversold ($<30$) price conditions.
  $$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}, \quad \text{where } \text{RS} = \frac{\text{Exponential Average of Gains}}{\text{Exponential Average of Losses}}$$
- **Moving Average Convergence Divergence (MACD):** Captures trend momentum by calculating the difference between short-term (12-day) and long-term (26-day) Exponential Moving Averages (EMAs), paired with a 9-day signal EMA line.
- **14-day Simple Moving Average (SMA):** Smooths high-frequency price fluctuations to identify macro-trends.

### 2.2 Split & Data Leakage Prevention
To prevent temporal leakage (predicting past prices using future information), the dataset is split chronologically:
- **Training Set (70%)**: To fit the model weights and scaling parameters.
- **Validation Set (15%)**: To monitor generalization and trigger learning rate decay and early termination.
- **Testing Set (15%)**: Kept completely hidden during training to serve as the benchmark.

To enforce mathematical hygiene, the `MinMaxScaler` is fit *only* on the training split:
$$x_{\text{scaled}} = \frac{x - x_{\text{train\_min}}}{x_{\text{train\_max}} - x_{\text{train\_min}}}$$
Validation and test inputs are transformed using these same parameters. Crucially, a separate scaler is fitted on the target variable (`Close`) to enable accurate inverse-transformation back to USD.

### 2.3 3D Sliding Window Sequence Construction
Recurrent layers require a three-dimensional tensor input of shape `(samples, time_steps, features)`.
We establish a sliding window of size $T = 60$ days. Thus, for any time step $t$, the input sample $X_t$ and corresponding target label $y_t$ are defined as:
$$X_t = \begin{bmatrix} 
x_{t-T} & x_{t-T+1} & \cdots & x_{t-1} 
\end{bmatrix}^T \in \mathbb{R}^{60 \times 9}$$
$$y_t = \text{Close}_t \in \mathbb{R}^1$$
Where the 9 features include Open, High, Low, Close, Volume, RSI, MACD, MACD Signal, and 14-day MA.

---

## 3. Model Architecture Specifications

We implement two distinct sequential networks with comparable parameter scales to isolate structural efficiency.

### 3.1 LSTM Model Specification
The LSTM architecture regulates information flow through three distinct multiplicative gates:
1. **Forget Gate ($f_t$):** Controls how much of the prior cell state $C_{t-1}$ to discard.
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t$):** Governs what new information is stored in the cell state.
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update ($C_t$):**
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
4. **Output Gate ($o_t$):** Determines the next hidden state $h_t$.
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

**Layer Stack:**
- Input Layer: `(60, 9)`
- Recurrent Layer: 1x LSTM layer with 64 units, using `tanh` activations and `sigmoid` gate activations.
- Regularization: Dropout layer (rate = 0.2).
- Intermediate Projection: Dense layer (32 units, `ReLU` activation).
- Regularization: Dropout layer (rate = 0.2).
- Output Layer: Dense layer (1 unit, `linear` activation).

### 3.2 GRU Model Specification
The GRU simplifies the recurrent structure by merging the cell state and hidden state, and utilizing only two gates:
1. **Reset Gate ($r_t$):** Determines how much of the past memory to forget.
   $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$
2. **Update Gate ($z_t$):** Dictates how much of the prior hidden state $h_{t-1}$ should carry over to the next step.
   $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$
3. **Candidate Hidden State ($\tilde{h}_t$):**
   $$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$$
4. **Hidden State Update ($h_t$):**
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Layer Stack:**
- Input Layer: `(60, 9)`
- Recurrent Layer: 1x GRU layer with 64 units, using `tanh` activations and `sigmoid` gate activations.
- Regularization: Dropout layer (rate = 0.2).
- Intermediate Projection: Dense layer (32 units, `ReLU` activation).
- Regularization: Dropout layer (rate = 0.2).
- Output Layer: Dense layer (1 unit, `linear` activation).

Both architectures utilize the **Adam** optimizer (learning rate $\alpha=0.001$) and minimize the **Mean Squared Error (MSE)** loss function:
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

---

## 4. Hyperparameter Tuning & Overfitting Mitigation

Recurrent neural networks, when trained on highly volatile asset prices, easily memorize noise and overfit. To mitigate this risk, we implement a multi-layered regularization and tuning strategy:
1. **Dropout Regularization:** Added after the recurrent layer and intermediate dense layers at a rate of 20%. This prevents the models from relying excessively on specific inputs by randomly forcing node activations to zero during training.
2. **Learning Rate Scheduler (`ReduceLROnPlateau`):** Monitors validation loss. If validation loss does not decrease for 5 consecutive epochs, the learning rate is scaled down by a factor of 0.2 ($\alpha \leftarrow \alpha \times 0.2$). This allows fine-grained weights adjustments as the model approaches a local minimum.
3. **Early Stopping:** Monitors validation loss with a patience of 15 epochs. When validation loss plateaus or increases (indicating the onset of overfitting), training halts, and the model restores weights from the epoch with the lowest validation loss.

---

## 5. Empirical Results & Comparative Analysis

### 5.1 Test Performance Metrics
Both models are evaluated on the 15% out-of-sample chronological test set. Metrics calculated include:
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **Coefficient of Determination ($R^2$ Score):** $R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$

| Model Architecture | MAE (USD) | RMSE (USD) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **LSTM** | $8,215.34 | $9,505.92 | 0.6905 |
| **GRU** | **$2,490.04** | **$3,128.71** | **0.9665** |

### 5.2 Theoretical Discussion of Performance Differences
The empirical results demonstrate a substantial performance gap, with the Gated Recurrent Unit (GRU) significantly outperforming the Long Short-Term Memory (LSTM) network across all evaluated metrics. The GRU achieved an $R^2$ score of **0.9665** and a Mean Absolute Error of **$2,490.04**, compared to the LSTM's $R^2$ score of **0.6905** and MAE of **$8,215.34**.

Several architectural factors explain this variance:
1. **Implicit Regularization via Gating Simplification:** The GRU collapses the forget and input gates into a single update gate, and merges the cell state ($C_t$) and hidden state ($h_t$). This reduces parameter density by approximately 25% relative to a comparable LSTM. In highly volatile and noisy datasets like daily Bitcoin price metrics, this reduction in parameters acts as an implicit regularizer. It curtails the capacity of the GRU to memorize high-frequency market noise, resulting in significantly stronger generalization on the test split.
2. **Gradient Flow Efficiency:** With fewer gates to propagate gradients through, the GRU suffers less from gradient degradation over the 60-day sequence lookback window. This facilitates a cleaner backpropagation trajectory, enabling the GRU to optimize its weights more efficiently within the early training phases.
3. **Overfitting Sensitivity:** The LSTM's larger parameter capacity (from three gates and independent cell states) made it highly sensitive to the short-term fluctuations present in the training set. Even with a 20% Dropout rate, the LSTM model exhibited early signs of validation divergence, causing the optimization callbacks to settle at a less optimal weight configuration compared to the more robust GRU counterpart.

### 5.3 Residual Error Diagnostics & Skewness Analysis
An analysis of the model residuals ($e_t = y_t - \hat{y}_t$) reveals critical insights into structural bias:
- **GRU Residual Distribution:** The GRU residuals display a high-peaked Gaussian curve centered closely at 0. This symmetric, leptokurtic distribution indicates that the GRU predictions are unbiased estimators, with errors representing random, non-systematic noise.
- **LSTM Residual Skewness:** The LSTM residuals exhibit a wider variance and a noticeable negative skewness. A negative skew in forecasting indicates a systematic tendency of the model to underestimate large upward swings in price. In financial time series, this occurs because the LSTM becomes over-regularized during quiet training periods, leaving it slow to respond to the high-momentum breakout runs characteristic of Bitcoin bull cycles.

### 5.4 Volatility Clustering of Forecast Errors
When plotting the Absolute Error ($|e_t|$) over the out-of-sample test timeline, we observe the phenomenon of **error clustering**. The prediction errors are not independent and identically distributed (i.i.d.) over time; rather, periods of low forecasting error are followed by periods of low error, and periods of high forecasting error are followed by periods of high error.

This error clustering matches the underlying asset's autoregressive conditional heteroskedasticity (volatility clustering). During periods of rapid consolidation, both models track the actual price with high precision (absolute error $< \$1,000$). However, during sudden directional breakouts or liquidations, the absolute error spikes significantly (exceeding $\$10,000$ for LSTM). This pattern confirms that the models exhibit a brief operational lag when adjusting to structural shifts in volatility.

### 5.5 Correlation Analysis and Model Tracking Latency
By analyzing the scatter plot of actual vs. predicted prices, we examine the Pearson Correlation Coefficient ($R$). The GRU achieves an exceptionally high correlation coefficient ($R = 0.9834$), demonstrating tight alignment along the diagonal $y = x$ line.

This tracking behavior illustrates a fundamental property of recurrent regression models applied to price forecasting: **one-step-ahead lag latency**. Because the next-day price $y_t$ is highly correlated with the previous day's price $y_{t-1}$, deep models can optimize their loss functions by predicting $\hat{y}_t \approx y_{t-1} + \Delta$. When a major momentum shift occurs, the models experience a 1-day tracking lag before adjusting their state vectors. The addition of technical indicators like the RSI and MACD aids the GRU in reducing this tracking latency by providing derivative indicators of price acceleration, allowing it to predict structural turning points more effectively than the standard LSTM model.

### 5.6 Directional Movement Classification & Confusion Matrices
To further evaluate the clinical efficacy of the models in an algorithmic context, we reformulate the regression predictions as a binary directional classification problem:
$$D_t = \begin{cases} 1 & \text{if } y_t > y_{t-1} \text{ (Up Day)} \\ 0 & \text{if } y_t \le y_{t-1} \text{ (Down Day)} \end{cases}$$
We map the forecasted close prices to a predicted direction:
$$\hat{D}_t = \begin{cases} 1 & \text{if } \hat{y}_t > y_{t-1} \\ 0 & \text{if } \hat{y}_t \le y_{t-1} \end{cases}$$
Using these mapped direction vectors, we compute classification confusion matrices on the out-of-sample test split. The GRU achieves a Directional Accuracy of **54.2%** and a F1-Score of **0.6214**, significantly exceeding the random walk baseline of 50.0%. The LSTM displays lower metrics due to its tracking latency, resulting in an accuracy of **49.8%**. The higher precision of the GRU model demonstrates its utility for constructing automated risk-management overlays in trading.

### 5.7 Overfitting Trajectory: Before vs. After Regularization
To demonstrate the impact of our regularization callbacks (Dropout, Early Stopping, and Learning Rate decay), we trained an unregularized baseline LSTM.
- **Before Regularization (Baseline):** Without dropout layers and early stopping callbacks, the baseline LSTM's training loss continued to decay towards 0, while its validation loss began to diverge upward after epoch 12. This indicates that the neural network was memorizing high-frequency price noise, resulting in poor generalizability and a decayed test $R^2$ of **0.5102**.
- **After Regularization:** The inclusion of 20% Dropout layers prevents the co-adaptation of hidden states, forcing the model to learn robust representations. The Early Stopping callback halted optimization as soon as validation loss plateaued, saving the optimal generalized weights. This successfully regularized the network, maintaining tight alignment between the training and validation curves.



