import csv

# Path to the info.csv file
info_csv_file = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\working\\info.csv'

# Path to the sentiment_eng.csv file
sentiment_csv_file = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\working\\sentiment_eng.csv'

# Path to the output info_filtered.csv file
output_csv_file = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\working\\info_filtered.csv'

# Read the ClassID values from the sentiment_eng.csv file
class_ids = set()
with open(sentiment_csv_file, 'r') as sentiment_file:
    reader = csv.DictReader(sentiment_file)
    for row in reader:
        class_ids.add(row['ClassID'])

# Filter the rows from the info.csv file based on ClassID
with open(info_csv_file, 'r') as info_file, open(output_csv_file, 'w', newline='') as output_file:
    reader = csv.DictReader(info_file)
    fieldnames = reader.fieldnames

    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        if row['ClassID'] in class_ids:
            writer.writerow(row)
