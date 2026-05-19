# Project Gold: Price Prediction & Signaling

This project provides a machine learning pipeline to predict gold price movements and generate investment signals (BUY, SELL, HOLD).

## Features
- **Data Collection**: Automatic download of historical gold futures data via `yfinance`.
- **Machine Learning**: XGBoost model trained to predict next-day returns with ~57% directional accuracy.
- **Technical Analysis**: Real-time calculation of RSI, MACD, Moving Averages, and Volatility.
- **Dashboard**: Interactive Streamlit app for visualization and decision support.

## Project Structure
- `data/`: Contains historical CSV data.
- `models/`: Trained XGBoost model and feature definitions.
- `notebooks/`: (Optional) For exploratory analysis.
- `collect_data.py`: Script to update historical data.
- `train_model.py`: Training pipeline for the XGBoost regressor.
- `app.py`: Streamlit dashboard.

## How to Run
1. Ensure dependencies are installed:
   ```bash
   pip install yfinance xgboost streamlit plotly joblib pandas numpy
   ```
2. (Optional) Re-train the model:
   ```bash
   python train_model.py
   ```
3. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## Investment Signals
- **INVEST (BUY)**: Predicted upward movement > 0.2%.
- **SELL / SHORT**: Predicted downward movement > 0.2%.
- **HOLD**: Predicted movement within +/- 0.2%.

**Disclaimer**: This is a prototype for educational and informational purposes. Financial markets are inherently risky. Never trade with money you cannot afford to lose.
