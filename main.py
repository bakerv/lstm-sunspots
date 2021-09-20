import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(tf.__version__)
#%%
# url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
# wget.download(url)
# print('daily-min-temperatures.csv')


#%% Data extraction
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


#%% data exploration
print(time_step)
print(min_temp)
print(len(min_temp))

#%% Split data for train and test
split_time = 2500
time_train = time[:split_time]
values_train = series[:split_time]
time_validation = time[split_time:]
values_validation = series[split_time:]

window_size = 30
batch_size = 32
shuffle_buffer_size = 1000


#%%
def window_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = (tf.data.Dataset.from_tensor_slices(series)
          .window(window_size + 1, shift=1, drop_remainder=True)
          .flat_map(lambda w: w.batch(window_size + 1))
          .shuffle(shuffle_buffer)
          .map(lambda w: (w[:-1], w[1:]))
          )
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    ds = (tf.data.Dataset.from_tensor_slices(series)
          .window(window_size, shift=1, drop_remainder=True)
          .flat_map(lambda w: w.batch(window_size))
          .batch(32).prefetch(1))
    forecast = model.predict(ds)
    return forecast


#%%
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # YOUR CODE HERE
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
#%%
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

train_set = windowed_dataset(values_train,
                             window_size,
                             batch_size,
                             shuffle_buffer_size)
print(train_set)
print(values_train.shape)

#%%
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                           strides=1, padding = "causal",
                           activation='relu',
                           input_shape=[None,1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100)
])

test_lr = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch /20))
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=sgd_optimizer,
              metrics=['mae'])
history = model.fit(train_set, epochs=100, callbacks=[test_lr])

#%%
print(history.history)

#%%
plt.semilogx(history.history['lr'], history.history['loss'])
plt.show()

#%%
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                           strides=1, padding = "causal",
                           activation='relu',
                           input_shape=[None,1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100)
])

sgd_optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=sgd_optimizer,
              metrics=['mae'])
history = model.fit(train_set, epochs=500)

#%%
plt.plot(history.epoch, history.history['loss'])
plt.show()

#%%

temp_forecast = (model_forecast(model,
                               series[...,np.newaxis],
                               window_size)
[split_time - window_size:-1, -1,  0])


#%%

plt.plot(time_validation, values_validation)
plt.plot(time_validation, temp_forecast)

#%%

tf.keras.metrics.mean_absolute_error(values_validation, temp_forecast).numpy()

