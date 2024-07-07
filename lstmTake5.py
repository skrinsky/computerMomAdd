import json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data_from_json(json_file):
    """Load sequences from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def lr_schedule(epoch):
    """Learning Rate Schedule"""
    lr = 0.001
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    return lr

def prepare_sequences(sequences, sequence_length=100):
    """Prepare input and output sequences for LSTM."""
    X = []
    y = []

    for sequence in sequences:
        if len(sequence) != sequence_length:
            print(f"Skipping sequence of length {len(sequence)}")
            continue

        pitches = []
        velocities = []
        start_times = []
        end_times = []

        for note in sequence:
            pitches.append(note['pitch'])
            velocities.append(note['velocity'])
            start_times.append(note['start_time'])
            end_times.append(note['end_time'])

        X.append((pitches, velocities, start_times, end_times))
        y.append(sequence[-1]['pitch'])

    X = np.array(X)
    print(f"Shape of X before reshaping: {X.shape}")
    X = np.reshape(X, (X.shape[0], sequence_length, 4))
    y = np.array(y)
    y = to_categorical(y)

    return X, y

def create_lstm_model(input_shape, output_shape):
    """Create LSTM model."""
    optimizer = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def train_lstm_model(X_train, y_train, X_val, y_val):
    """Train LSTM model."""
    model = create_lstm_model(input_shape=X_train.shape[1:], output_shape=y_train.shape[1])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, lr_scheduler])

    return model, history

def plot_training_history(history):
    """Plot training and validation loss and accuracy."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.show()

def main():
    json_file = 'data/100_note_sequences_by_song.json'
    sequences = load_data_from_json(json_file)
    
    print(f"Number of sequences loaded: {len(sequences)}")
    for i, seq in enumerate(sequences[:5]):
        print(f"Sequence {i} length: {len(seq)}")
    
    X, y = prepare_sequences(sequences)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    lstm_model, history = train_lstm_model(X_train, y_train, X_val, y_val)

    lstm_model.save('lstm_model.h5')
    plot_training_history(history)

if __name__ == '__main__':
    main()
