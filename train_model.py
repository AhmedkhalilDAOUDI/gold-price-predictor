import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def calculate_indicators(df):
    # Base Indicators
    df['SMA_7_Rel'] = df['Gold'] / df['Gold'].rolling(window=7).mean()
    df['SMA_20_Rel'] = df['Gold'] / df['Gold'].rolling(window=20).mean()
    df['SMA_50_Rel'] = df['Gold'] / df['Gold'].rolling(window=50).mean()
    
    # RSI for Gold
    delta = df['Gold'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['Volatility'] = df['Gold'].pct_change().rolling(window=7).std()
    
    # Macro Interactions (Relative strength vs other assets)
    df['Gold_vs_SP500'] = df['Gold'] / df['SP500']
    df['Gold_vs_USD'] = df['Gold'] / df['USD_Index']
    
    # Daily returns for all assets to capture correlations
    assets = ['Gold', 'USD_Index', 'SP500', '10Y_Yield', 'Oil', 'Silver']
    for asset in assets:
        df[f'{asset}_Ret'] = df[asset].pct_change()
        # Lags
        for i in range(1, 4):
            df[f'{asset}_Lag_{i}'] = df[asset].pct_change(periods=i)
            
    return df

def train_gold_model():
    print("Loading macro-enhanced data...")
    df = pd.read_csv("data/market_data_macro.csv", parse_dates=True, index_col=0)
    df.sort_index(inplace=True)
    
    df = calculate_indicators(df)
    
    # Target: Predict Next Day Return
    df['Target_Return'] = df['Gold'].pct_change().shift(-1)
    
    df.dropna(inplace=True)
    
    # Split (Time-aware)
    train_size = int(len(df) * 0.9)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Features
    features = [
        'SMA_7_Rel', 'SMA_20_Rel', 'SMA_50_Rel', 'RSI', 'Volatility', 
        'Gold_vs_SP500', 'Gold_vs_USD'
    ] + [f'{a}_Ret' for a in ['Gold', 'USD_Index', 'SP500', '10Y_Yield', 'Oil', 'Silver']] \
      + [f'Gold_Lag_{i}' for i in range(1, 4)] \
      + [f'USD_Index_Lag_{i}' for i in range(1, 4)]
    
    X_train, y_train = train_df[features], train_df['Target_Return']
    X_test, y_test = test_df[features], test_df['Target_Return']
    
    print(f"Training on {len(X_train)} samples...")
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )
    
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)
    
    # Evaluation
    predictions = model.predict(X_test)
    dir_true = np.sign(y_test)
    dir_pred = np.sign(predictions)
    accuracy = (dir_true == dir_pred).mean()
    
    print(f"Directional Accuracy (Macro-Enhanced): {accuracy:.2%}")
    
    # Save
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, "models/gold_model_v2.pkl")
    joblib.dump(features, "models/features_v2.pkl")
    print("Macro-enhanced model saved successfully.")

if __name__ == "__main__":
    train_gold_model()
