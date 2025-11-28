# src/reporting.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_report(df, output_folder):
    """
    Generates a CSV report with detailed results and a bar chart visualization.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Save the detailed results to a CSV file
    detailed_csv_path = os.path.join(output_folder, 'detailed_sentiment_analysis.csv')
    df.to_csv(detailed_csv_path, index=False)
    print(f"Detailed analysis saved to: {detailed_csv_path}")

    # 2. Create and save a sentiment distribution bar chart
    sentiment_counts = df['sentiment'].value_counts()
    
    # --- START OF MODIFICATION ---

    # Define a color map to ensure consistent colors for each sentiment
    color_map = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'grey'
    }

    # Create a list of colors in the same order as the sentiment_counts index
    # Use .get() to provide a default color (e.g., 'blue') if a sentiment is not in the map
    colors_for_plot = [color_map.get(sentiment, 'blue') for sentiment in sentiment_counts.index]

    plt.figure(figsize=(10, 7)) # Made the figure a bit bigger for clarity
    
    # Use the new color list for plotting
    sentiment_counts.plot(kind='bar', color=colors_for_plot)
    
    # --- END OF MODIFICATION ---

    plt.title('Sentiment Analysis Distribution', fontsize=16)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Comments', fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Added a grid for easier reading
    
    chart_path = os.path.join(output_folder, 'sentiment_distribution.png')
    plt.savefig(chart_path, bbox_inches='tight') # Added bbox_inches for better layout
    plt.close() # Close the plot to free up memory

    print(f"Sentiment distribution chart saved to: {chart_path}")