import pandas as pd
import json

# Parameters
num_samples = 2000  # Number of random records
dataset_name = "Lenta.ru news"

# Read CSV file and parse date
df = pd.read_csv("lenta-ru-news.csv", parse_dates=["date"], dayfirst=False)

# Filter by date range
df_filtered = df[(df["date"] >= "2010-01-01") & (df["date"] <= "2019-12-31")]

# Select random records
df_sampled = df_filtered.sample(n=min(num_samples, len(df_filtered)), random_state=42)

# Convert to desired format
records = [
    {
        "id": i + 1,
        "text": row.text,  # Access text field via attribute
        "source": "human",
        "dataset": dataset_name
    }
    for i, row in enumerate(df_sampled.itertuples(index=False, name="Record"))
]

# Save to JSON with desired indentation
with open("filtered_data.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=4)

print(f"File saved: filtered_data.json with {num_samples} random records.")
