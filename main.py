import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wget

print(tf.__version__)


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Sunspots')
    plt.grid(True)


# url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
# wget.download(url)
# print('daily-min-temperatures.csv')

time_step = []
min_temp = []
with open ('daily-min-temperatures.csv') as temp:
    reader = csv.reader(temp, delimiter=',')
    next(reader)
    steps = 1
    for row in reader:
        sequence = steps
        temp = float(row[1])
        time_step.append(sequence)
        min_temp.append(temp)
        steps += 1

series = np.array(min_temp)
time = np.array(time_step)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

print(time_step)
print(min_temp)


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
window_size = 64
batch_size = 256
