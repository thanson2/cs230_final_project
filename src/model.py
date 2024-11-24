#!pip install yfinance
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D

import yfinance as yf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

def calculate_bollinger_bands(data, window=10, num_of_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def calculate_rsi(data, window=10):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_roc(data, periods=10):
    """Calculate Rate of Change."""
    roc = ((data - data.shift(periods)) / data.shift(periods)) * 100
    return roc

#TODO add more

#tickers = ['META', 'AAPL', 'MSFT', 'AMZN', 'GOOG']

s_and_p = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
#print(s_and_p.head())
tickers = [symbol for symbol in s_and_p.Symbol.to_list() if str.isalpha(symbol)]

# data = yf.download(tickers.Symbol.to_list(),'2021-1-1','2021-7-12', auto_adjust=True)['Close']
# print(data.head())


# f = yf.download('META', period="1y", interval="1h")
# print(f)

y = yf.download('AMTM', period="1y", interval="1h")
print(y.empty)

# msft = yf.Ticker("AMTM")
# msft.info
# msft.quarterly_income_stmt

#TODO normalize all of the inputs

ticker_data_frames = []
stats = {}

remove = []
for ticker in tickers:

    # Download historical data for the ticker
    data = yf.download(ticker, period="1mo", interval="5m") #TODO stated at 1mo, 5m, could have hyperparam search on this too
    if data.empty:
      remove.append(ticker)
      continue

    # Calculate the daily percentage change
    close = data['Close'] #just the daily close
    upper, lower = calculate_bollinger_bands(close, window=14, num_of_std=2)
    width = upper - lower
    rsi = calculate_rsi(close, window=14)
    roc = calculate_roc(close, periods=14)
    volume = data['Volume']
    diff = data['Close'].diff(1)
    percent_change_close = data['Close'].pct_change() * 100

    width = width.squeeze()
    rsi = rsi.squeeze()
    roc = roc.squeeze()
    diff = diff.squeeze()
    percent_change_close = percent_change_close.squeeze()
    volume = volume.squeeze()
    diff = diff.squeeze()
    percent_change_close = percent_change_close.squeeze()
    close = close.squeeze()

    #TODO expirement with different metrics
    ticker_df = pd.DataFrame({
        ticker+'_close': close,
        ticker+'_width': width,
        ticker+'_rsi': rsi,
        ticker+'_roc': roc,
        ticker+'_volume': volume,
        ticker+'_diff': diff,
        ticker+'_percent_change_close': percent_change_close,
    })

    MEAN = ticker_df.mean()
    STD = ticker_df.std()

    # Keep track of mean and std
    for column in MEAN.index:
      stats[f"{column}_mean"] = MEAN[column]
      stats[f"{column}_std"] = STD[column]

    # Normalize the training features
    ticker_df = (ticker_df - MEAN) / STD

    ticker_data_frames.append(ticker_df)

for ticker in remove:
  tickers.remove(ticker)

# Convert stats from dict to df
stats = pd.DataFrame([stats], index=[0])
stats.head()

# Concatenate all ticker DataFrames
df = pd.concat(ticker_data_frames, axis=1)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.head()

# Shift the df data to create labels
labels = df.shift(-1)

# Drop the last row in both percent_change_data and labels as it won't have a corresponding label
df = df.iloc[:-1]
labels = labels.iloc[:-1]

# Sequence len = 24 means that we have 2 hours of 5 min data
SEQUENCE_LEN = 24

# Function to create X-day sequences for each ticker
def create_sequences(data, labels, mean, std, sequence_length=SEQUENCE_LEN):
    sequences = []
    lab = []
    data_size = len(data)

    # -1 because we want to know the last close of the sequence.
    # + 12 because we want to predict the next hour
    for i in range(1, data_size - (sequence_length + 13)):
      sequences.append(data[i:i + sequence_length])
      lab.append([labels[i-1], labels[i + 12], mean[0], std[0]])

    return np.array(sequences), np.array(lab)

sequences_dict = {}
sequence_labels = {}
for ticker in tickers:

    # Extract close and volume data for the ticker
    close = df[ticker+'_close'].values
    width = df[ticker+'_width'].values
    rsi = df[ticker+'_rsi'].values
    roc = df[ticker+'_roc'].values
    volume = df[ticker+'_volume'].values
    diff = df[ticker+'_diff'].values
    pct_change = df[ticker+'_percent_change_close'].values

    # Combine close and volume data
    ticker_data = np.column_stack((close,
                                   width,
                                   rsi,
                                   roc,
                                   volume,
                                   diff,
                                   pct_change))

    # Generate sequences
    attribute = ticker+"_close"
    ticker_sequences, lab = create_sequences(ticker_data,
                                             labels[attribute].values[SEQUENCE_LEN-1:],
                                             stats[attribute+"_mean"].values,
                                             stats[attribute+"_std"].values)

    sequences_dict[ticker] = ticker_sequences
    sequence_labels[ticker] = lab

# Combine data and labels from all tickers

all_sequences = []
all_labels = []

for ticker in tickers:
    all_sequences.extend(sequences_dict[ticker])
    all_labels.extend(sequence_labels[ticker])

# Convert to numpy arrays
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)
print(f"{all_sequences.shape=}")
print(f"{all_labels.shape=}")

'''


# Combine data and labels from all tickers

TRAIN_FRACTION = 0.8

train_sequences = []
train_labels = []
other_sequences = []
other_labels = []

def extend_sequences(ticker):
  ticker_sequence = sequences_dict[ticker]
  ticker_labels = sequence_labels[ticker]
  seq_len = len(ticker_sequence)
  print(seq_len)
  assert seq_len == len(ticker_labels)
  train_len = int(seq_len * TRAIN_FRACTION)
  train_sequences.extend(ticker_sequence[:train_len])
  train_labels.extend(ticker_labels[:train_len])
  # Leave a gap between the train and other splits so they don't overlap in time.
  other_sequences.extend(ticker_sequence[train_len + SEQUENCE_LEN:])
  other_labels.extend(ticker_labels[train_len + SEQUENCE_LEN:])


for ticker in tickers:
    extend_sequences(ticker)

# Convert to numpy arrays
train_sequences = np.array(train_sequences)
train_labels = np.array(train_labels)
other_sequences = np.array(other_sequences)
other_labels = np.array(other_labels)

print(f"{train_sequences.shape=}")
print(f"{train_labels.shape=}")
print(f"{other_sequences.shape=}")
print(f"{other_labels.shape=}")
'''

np.random.seed(42)
shuffled_indices = np.random.permutation(len(all_sequences))
all_sequences = all_sequences[shuffled_indices]
all_labels = all_labels[shuffled_indices]

train_size = int(len(all_sequences) * 0.9)

# Split sequences
train_sequences = all_sequences[:train_size]
train_labels    = all_labels[:train_size]

other_sequences = all_sequences[train_size:]
other_labels    = all_labels[train_size:]

shuffled_indices = np.random.permutation(len(other_sequences))
other_sequences = other_sequences[shuffled_indices]
other_labels = other_labels[shuffled_indices]

val_size = int(len(other_sequences) * 0.5)

validation_sequences = other_sequences[:val_size]
validation_labels = other_labels[:val_size]

test_sequences = other_sequences[val_size:]
test_labels = other_labels[val_size:]

'''
np.random.seed(42)
print(len(other_sequences))

# Randomize the order of the training data
shuffled_indices = np.random.permutation(len(train_sequences))
train_sequences = train_sequences[shuffled_indices]
train_labels = train_labels[shuffled_indices]

# Validation/Test split is 50/50
val_size = int(len(other_sequences) * 0.5)

validation_sequences = other_sequences[:val_size]
validation_labels = other_labels[:val_size]

# Leave a gap between the val and test splits so they don't overlap in time.
test_sequences = other_sequences[val_size + SEQUENCE_LEN:]
test_labels = other_labels[val_size + SEQUENCE_LEN:]

print(f"{validation_sequences.shape=}")
print(f"{validation_labels.shape=}")
print(f"{test_sequences.shape=}")
print(f"{test_labels.shape=}")
'''

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Add()([x, inputs])

    # Feed Forward Part
    y = LayerNormalization(epsilon=1e-6)(x)
    y = Dense(ff_dim, activation="relu")(y)
    y = Dropout(dropout)(y)
    y = Dense(inputs.shape[-1])(y)
    return Add()([y, x])

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers, dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs

    # Create multiple layers of the Transformer block
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Final part of the model
    x = GlobalAveragePooling1D()(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    outputs = Dense(1, activation="linear")(x) #TODO coukd try different activations

    # Compile model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def custom_mae_loss(y_true, y_pred):
    y_true_next = tf.cast(y_true[:, 1], tf.float64)
    y_pred_next = tf.cast(y_pred[:, 0], tf.float64)
    abs_error = tf.abs(y_true_next - y_pred_next)

    return tf.reduce_mean(abs_error)

#TODO expirament with different loss functions

def dir_acc(y_true, y_pred):
    mean, std = tf.cast(y_true[:, 2], tf.float64), tf.cast(y_true[:, 3], tf.float64)

    y_true_prev = (tf.cast(y_true[:, 0], tf.float64) * std) + mean
    y_true_next = (tf.cast(y_true[:, 1], tf.float64) * std) + mean
    y_pred_next = (tf.cast(y_pred[:, 0], tf.float64) * std) + mean

    true_change = y_true_next - y_true_prev
    pred_change = y_pred_next - y_true_prev

    correct_direction = tf.equal(tf.sign(true_change), tf.sign(pred_change))

    return tf.reduce_mean(tf.cast(correct_direction, tf.float64))

#TODO add absolute accuracy, or metric based on how much you would have made on a trade?

# Define a callback to save the best model
checkpoint_callback_train = ModelCheckpoint(
    "transformer_train_model.keras",  # Filepath to save the best model
    monitor="dir_acc",  #"loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="max",  # Minimize the monitored metric
    verbose=1,  # Display progress
)

# Define a callback to save the best model
checkpoint_callback_val = ModelCheckpoint(
    "transformer_val_model.keras",  # Filepath to save the best model
    monitor="val_dir_acc", #"val_loss",  # Metric to monitor
    save_best_only=True,  # Save only the best model
    mode="max",  # Minimize the monitored metric
    verbose=1,  # Display progress
)

'''
def get_lr_callback(batch_size=16, mode='cos', epochs=500, plot=False): #TODO is this needed with Adam?
    lr_start, lr_max, lr_min = 0.0001, 0.005, 0.00001  # Adjust learning rate boundaries
    lr_ramp_ep = int(0.30 * epochs)  # 30% of epochs for warm-up
    lr_sus_ep = max(0, int(0.10 * epochs) - lr_ramp_ep)  # Optional sustain phase, adjust as needed

    def lrfn(epoch):
        if epoch < lr_ramp_ep:  # Warm-up phase
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:  # Sustain phase at max learning rate
            lr = lr_max
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        else:
            lr = lr_min  # Default to minimum learning rate if mode is not recognized

        return lr

    if plot:  # Plot learning rate curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Scheduler')
        plt.show()

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    '''

import random
import os

EPOCHS = 100
# Model parameters
input_shape = train_sequences.shape[1:]
head_size_arr = [64, 128, 256, 512]#, 1024] #256
num_heads_arr = [8, 16, 32, 64] #16
ff_dim_arr = [256, 512, 1024]#, 2048] # 1024
num_layers_arr = [8, 10, 12, 14, 16, 18] # 12
dropout_arr = [0, 0.05, 0.10, 0.20, 0.3, 0.4, 0.5, 0.6] # 0.20
batch_size_arr = [32, 64, 128]

best_accuracy = 0
best_params = None

with open('docs/hyperparams.txt', 'w') as f:
  f.write('START OF HYPERPARAM SEARCH')
  for i in range(20):
    try:
      if os.path.exists("transformer_train_model.keras"):
        os.remove("transformer_train_model.keras")
      if os.path.exists("transformer_val_model.keras"):
        os.remove("transformer_val_model.keras")
      batch_size = random.choice(batch_size_arr)
      head_size = random.choice(head_size_arr)
      num_heads = random.choice(num_heads_arr)
      ff_dim = random.choice(ff_dim_arr)
      num_layers = random.choice(num_layers_arr)
      dropout = random.choice(dropout_arr)

      # Build the model
      model = build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers, dropout)
      optimizer = tf.keras.optimizers.Adam()
      model.compile(optimizer=optimizer, loss=custom_mae_loss, metrics=[dir_acc])
      #model.summary()
      model.fit(train_sequences, train_labels,
                validation_data=(validation_sequences, validation_labels),
                epochs=EPOCHS,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[checkpoint_callback_train, checkpoint_callback_val]) #TODO removed get_lr_callback(batch_size=batch_size, epochs=EPOCHS)

      if os.path.exists("transformer_val_model.keras"):
        model.load_weights("transformer_val_model.keras")

      # Make predictions
      accuracy = model.evaluate(test_sequences, test_labels)[1]


      # Calculate additional metrics as needed
      from sklearn.metrics import r2_score

      predictions = model.predict(test_sequences)
      r2 = r2_score(test_labels[:, 1], predictions[:, 0])

      if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = (input_shape, head_size, num_heads, ff_dim, num_layers, dropout)

      write = f"R-squared: {r2}, accuracy: {accuracy}, batch_size: {batch_size}, head_size: {head_size}, num_heads: {num_heads}, ff_dim: {ff_dim}, num_layers: {num_layers}, dropout: {dropout}"
      print(write)
      f.write(write + "\n")
      f.flush()
    except Exception as e:
      print(f"Exception: {e}, continuing")

  best = f"best params: {best_params}, best accuracy: {best_accuracy}"
  print(best)
  f.write(best + "\n")
  f.flush()