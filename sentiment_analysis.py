import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re

# Load CSV file
df = pd.read_csv("amazon_reviews.csv")

# Print dataset
print("\nDATASET:")
print(df)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', str(text))
    return text.lower()

# Apply cleaning
df['Cleaned_Review'] = df['Review'].apply(clean_text)

# Sentiment function
def get_sentiment(text):

    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0:
        return "Positive"

    elif analysis.sentiment.polarity < 0:
        return "Negative"

    else:
        return "Neutral"

# Apply sentiment analysis
df['Sentiment'] = df['Cleaned_Review'].apply(get_sentiment)

# Final output
print("\nFINAL RESULT:")
print(df[['Review', 'Sentiment']])

# Graph
sns.countplot(x='Sentiment', data=df)

plt.title("Amazon Review Sentiment Analysis")

plt.show()
with open("output.txt", "w") as f:
    f.write(str(df[['Review','Sentiment']]))