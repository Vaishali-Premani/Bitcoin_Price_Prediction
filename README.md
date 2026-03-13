# ₿ Bitcoin Price Prediction using LSTM

A machine learning project that predicts **Bitcoin prices using Long Short-Term Memory (LSTM)** neural networks and provides an **interactive Streamlit dashboard** for visualization and analysis.

The system analyzes historical Bitcoin market data and forecasts the **next-day price**, helping users explore trends and understand cryptocurrency market behavior.

---

# 📌 Project Overview

Cryptocurrency markets are highly volatile and difficult to predict. Traditional statistical models struggle to capture complex time-series dependencies.

This project uses **Deep Learning (LSTM)** to learn patterns from historical Bitcoin data and predict future prices. The system also provides an interactive **web interface using Streamlit** for exploring data, visualizing trends, and viewing predictions.

---

# 🎯 Objectives

- Predict **Bitcoin price movements** using an LSTM model.
- Provide **next-day price predictions** based on historical market data.
- Visualize **historical trends and predicted values** interactively.
- Compare **Actual vs Predicted prices** for model evaluation.
- Demonstrate the use of **Deep Learning for financial time-series forecasting**.

---

# 🧠 Technologies Used

| Technology           | Purpose                    |
|----------------------|----------------------------|
| Python               | Programming Language       |
| TensorFlow / Keras   | LSTM Model Development     |
| Pandas               | Data Processing            |
| NumPy                | Numerical Computation      |
| Matplotlib / Plotly  | Data Visualization         |
| Streamlit            | Interactive Web Dashboard  |
| yfinance             | Bitcoin Data Collection    |

---

# ⚙️ Methodology

The workflow of the project includes the following steps:

### 1️⃣ Data Collection
Historical Bitcoin price data is collected using the **yfinance API**.

### 2️⃣ Data Preprocessing
- Handle required features
- Normalize data using **MinMaxScaler**
- Create **60-day sliding windows** for time-series learning

### 3️⃣ Model Development
- LSTM Neural Network is used
- Trained on historical data to capture temporal patterns
- Model saved as `lstm_bitcoin_model.h5`

### 4️⃣ Prediction
The trained model predicts the **next day's Bitcoin price** based on the previous 60 days.

### 5️⃣ Visualization
Interactive plots are generated to show:

- Historical Bitcoin price trends
- Actual vs Predicted prices
- Market insights

---

### Project Workflow

- Historical Bitcoin Data  
↓  
- Data Preprocessing  
↓  
- 60-Day Sliding Window Creation  
↓  
- LSTM Model Training  
↓  
- Price Prediction  
↓  
- Streamlit Dashboard Visualization
