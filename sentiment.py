import urllib.request
import xml.etree.ElementTree as ET
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def get_gold_sentiment():
    print("Fetching gold market news...")
    # RSS Feed from a financial source (using Reuters or Investing.com style feeds if available)
    # Using a reliable financial RSS feed
    rss_url = "https://www.investing.com/rss/news_95.rss" # Commodities news
    
    try:
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        sid = SentimentIntensityAnalyzer()
        
        news_items = []
        sentiments = []
        
        for item in root.findall('./channel/item'):
            title = item.find('title').text
            if 'gold' in title.lower():
                score = sid.polarity_scores(title)['compound']
                news_items.append(title)
                sentiments.append(score)
        
        if not sentiments:
            return "Neutral", 0.0, ["No specific gold news found in recent feeds."]
            
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        if avg_sentiment > 0.05:
            label = "Bullish (Positive)"
        elif avg_sentiment < -0.05:
            label = "Bearish (Negative)"
        else:
            label = "Neutral"
            
        return label, avg_sentiment, news_items[:5]
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return "Error", 0.0, [f"Could not fetch news: {e}"]

if __name__ == "__main__":
    label, score, news = get_gold_sentiment()
    print(f"Sentiment: {label} ({score})")
    print("Recent Headlines:")
    for n in news:
        print(f"- {n}")
