import csv

with open('dataset/IMDB Dataset.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    data = [i for i in reader]
training = data[:25000]
testing = data[25000:]

with open('dataset/training.csv', 'w') as train, open('dataset/testing.csv', 'w') as test:
    train_writer = csv.writer(train, delimiter=',')
    test_writer = csv.writer(test, delimiter=',')

    train_writer.writerows(training), test_writer.writerows(testing)
