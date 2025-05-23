import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
df = pd.read_csv('/app/data/truckerpath_1star_3620.csv')

# Check the column names and first few rows
print(df.columns)
print(df.head())

# Get info about data types and missing values
df.info()

# Check for nulls in each column
df.isnull().sum()

# Drop rows with no review content
df = df.dropna(subset=['content'])

# Convert date columns to datetime
df['at'] = pd.to_datetime(df['at'])
df['repliedAt'] = pd.to_datetime(df['repliedAt'], errors='coerce')  # some may be NaT

# Fill missing values in optional columns (safe checks for each column)
optional_columns = ['reviewCreatedVersion', 'appVersion', 'replyContent']
for column in optional_columns:
    if column in df.columns:
        df[column] = df[column].fillna('Unknown' if column != 'replyContent' else 'No Reply')
    else:
        print(f"'{column}' column not found!")

# (Optional) Drop columns you don't need
if 'userImage' in df.columns:
    df = df.drop(columns=['userImage'])

# Reset index
df.reset_index(drop=True, inplace=True)

print("Cleaning complete. Here's a preview:")
print(df.head())

# Check again after cleaning
# Get info about data types and missing values
df.info()

# Check for nulls in each column
df.isnull().sum()

# Sort reviews by thumbsUpCount (descending)
top_helpful_reviews = df.sort_values(by='thumbsUpCount', ascending=False)

# Display the top 500 most "helpful" 1-star reviews
print(top_helpful_reviews[['userName', 'content', 'thumbsUpCount', 'at']].head(500))

# Plot top 10 helpful reviews
top_10 = top_helpful_reviews.head(10)

def save_visualization(df):
    """Save visualization of top reviews."""
    try:
        output_dir = Path('/app/data')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        top_10 = df.head(10)
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_10['userName'], top_10['thumbsUpCount'], color='crimson')
        plt.xlabel('Thumbs Up Count')
        plt.title('Top 10 Most Helpful 1-Star Reviews')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_path = output_dir / 'top_10_most_helpful_reviews.png'
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")

save_visualization(top_helpful_reviews)

# Get the top 500 reviews sorted by thumbsUpCount
top_500_reviews = df.sort_values(by='thumbsUpCount', ascending=False).head(500)

# Export to CSV
top_500_reviews.to_csv('/app/data/top_500_1star_reviews.csv', index=False)

print("Top 500 1-star reviews with most likes exported successfully!")
