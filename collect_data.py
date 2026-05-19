import yfinance as yf
import pandas as pd
import os

def download_data():
    print("Downloading comprehensive market data...")
    
    # Symbols to track
    symbols = {
        "Gold": "GC=F",
        "USD_Index": "DX-Y.NYB",
        "SP500": "^GSPC",
        "10Y_Yield": "^TNX",
        "Oil": "CL=F",
        "Silver": "SI=F"
    }
    
    data_frames = []
    
    for name, ticker in symbols.items():
        print(f"Fetching {name} ({ticker})...")
        try:
            # Using 10y instead of max to ensure better overlap
            df = yf.download(ticker, period="10y")
            if df.empty:
                print(f"Warning: No data for {name}")
                continue
            # Handle MultiIndex if yfinance returns it
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Close']].rename(columns={'Close': name})
            data_frames.append(df)
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            
    if not data_frames:
        print("No data collected.")
        return
        
    # Join all data on index (Date)
    final_df = data_frames[0]
    for df in data_frames[1:]:
        final_df = final_df.join(df, how='inner')
    
    # Drop rows with any NaN to ensure consistency
    final_df.dropna(inplace=True)
        
    # Save
    if not os.path.exists("data"):
        os.makedirs("data")
    
    output_path = "data/market_data_macro.csv"
    final_df.to_csv(output_path)
    print(f"Comprehensive data saved to {output_path}. Shape: {final_df.shape}")

if __name__ == "__main__":
    download_data()
