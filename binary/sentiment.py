import csv
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Define a function to perform sentiment analysis on a word


def get_sentiment_score(word):
    sentiment_scores = sia.polarity_scores(word)
    compound_score = sentiment_scores['compound']
    return compound_score


# Path to your input CSV file
input_csv_file = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\working\\classes.csv'

# Path to the output CSV file
output_csv_file = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\working\\sentiment_eng.csv'

# Open the input and output CSV files
with open(input_csv_file, 'r') as input_file, open(output_csv_file, 'w', newline='') as output_file:
    reader = csv.DictReader(input_file)

    # Define the headers for the output CSV file
    fieldnames = ['ClassID', 'Word', 'SentimentScore']

    # Create a CSV writer object
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over each row in the input CSV file
    for row in reader:
        class_id = row['ClassName_eng']

        # Perform sentiment analysis on each word in the ClassID
        words = class_id.split()
        sentiment_scores = [get_sentiment_score(word) for word in words]

        # Store word-score pairs in the list
        word_scores = list(zip(words, sentiment_scores))

        # Sort the word-score pairs based on the sentiment score in decreasing order
        sorted_word_scores = sorted(
            word_scores, key=lambda x: x[1], reverse=True)

        # Write the sorted word-score pairs to the output CSV file
        for word, score in sorted_word_scores:
            if score >= 0.05 or score <= -0.05:
                writer.writerow(
                    {'ClassID': row['ClassID'], 'Word': word, 'SentimentScore': score})
