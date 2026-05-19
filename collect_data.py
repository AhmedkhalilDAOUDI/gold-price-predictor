import yfinance as yf
import pandas as pd
import os

def download_gold_data():
    print("Downloading historical gold data (GC=F)...")
    # GC=F is Gold Futures on Yahoo Finance
    gold = yf.Ticker("GC=F")
    
    # Get max historical data
    df = gold.history(period="max")
    
    if df.empty:
        print("Failed to download data.")
        return
    
    # Save to data folder
    output_path = "data/gold_prices.csv"
    df.to_csv(output_path)
    print(f"Data saved to {output_path}. Shape: {df.shape}")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    download_gold_data()
