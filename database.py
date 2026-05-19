import sqlite3
import pandas as pd
import os

DB_PATH = "data/portfolio.db"

def init_db():
    if not os.path.exists("data"):
        os.makedirs("data")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Table for trades
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  type TEXT,
                  amount REAL,
                  price REAL,
                  total REAL,
                  currency TEXT,
                  karat TEXT,
                  time TEXT)''')
    
    # Table for portfolio state
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                 (id INTEGER PRIMARY KEY,
                  balance REAL,
                  holdings REAL)''')
    
    # Initialize portfolio if empty
    c.execute("SELECT COUNT(*) FROM portfolio")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO portfolio (id, balance, holdings) VALUES (1, 100000.0, 0.0)")
    
    conn.commit()
    conn.close()

def get_portfolio():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT balance, holdings FROM portfolio WHERE id = 1", conn)
    conn.close()
    return df.iloc[0]['balance'], df.iloc[0]['holdings']

def update_portfolio(balance, holdings):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE portfolio SET balance = ?, holdings = ? WHERE id = 1", (balance, holdings))
    conn.commit()
    conn.close()

def add_trade(trade_type, amount, price, total, currency, karat, time):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO trades (type, amount, price, total, currency, karat, time) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (trade_type, amount, price, total, currency, karat, time))
    conn.commit()
    conn.close()

def get_trade_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT type, amount, price, total, currency, karat, time FROM trades ORDER BY id DESC", conn)
    conn.close()
    return df

def reset_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM trades")
    c.execute("UPDATE portfolio SET balance = 100000.0, holdings = 0.0 WHERE id = 1")
    conn.commit()
    conn.close()
