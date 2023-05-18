import csv

# Path to the info.csv file
info_csv_file = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\binary\\info.csv'

# Path to the sentiment_eng.csv file
sentiment_csv_file = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\working\\sentiment_eng.csv'

# Path to the output info_filtered.csv file
output_csv_file = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\binary\\info_filtered.csv'

# Read the ClassID values and sentiment scores from the sentiment_eng.csv file
class_sentiments = {}
with open(sentiment_csv_file, 'r') as sentiment_file:
    reader = csv.DictReader(sentiment_file)
    for row in reader:
        class_id = row['ClassID']
        sentiment_score = float(row['SentimentScore'])
        class_sentiments[class_id] = sentiment_score

# Filter the rows from the info.csv file based on ClassID and add the Negativity column
with open(info_csv_file, 'r') as info_file, open(output_csv_file, 'w', newline='') as output_file:
    reader = csv.DictReader(info_file)
    fieldnames = reader.fieldnames + ['Negativity']

    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        class_id = row['ClassID']
        if class_id in class_sentiments:
            sentiment_score = class_sentiments[class_id]
            if sentiment_score < 0:
                row['Negativity'] = '1'
            else:
                row['Negativity'] = '0'
            writer.writerow(row)
