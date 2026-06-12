import os
import pickle
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping

# ==================================================
# Load Dataset
# ==================================================

with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Use first 500k characters for faster training
text = text[:500000]

# ==================================================
# Tokenization
# ==================================================

tokenizer = Tokenizer()

tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print(f"Vocabulary Size: {total_words}")

# ==================================================
# Create N-Gram Sequences
# ==================================================

input_sequences = []

for line in text.split("\n"):

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        input_sequences.append(token_list[: i + 1])

print("Total Sequences:", len(input_sequences))

# ==================================================
# Padding
# ==================================================

max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_seq_len,
        padding="pre"
    )
)

print("Max Sequence Length:", max_seq_len)

# ==================================================
# Features & Labels
# ==================================================

X = input_sequences[:, :-1]

y = input_sequences[:, -1]

print("X Shape:", X.shape)
print("y Shape:", y.shape)

# ==================================================
# Build Model
# ==================================================

model = Sequential([

    Embedding(
        input_dim=total_words,
        output_dim=200,
        input_shape=(max_seq_len - 1,)
    ),

    LSTM(
        256,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(128),

    Dropout(0.2),

    Dense(
        total_words,
        activation="softmax"
    )

])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==================================================
# Training
# ==================================================

early_stop = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X,
    y,
    epochs=30,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

# ==================================================
# Save Files
# ==================================================

os.makedirs("models", exist_ok=True)

model.save("models/lstm_model.keras")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/max_seq_len.pkl", "wb") as f:
    pickle.dump(max_seq_len, f)

print("\nTraining Completed Successfully!")

print("\nSaved Files:")
print("models/lstm_model.keras")
print("models/tokenizer.pkl")
print("models/max_seq_len.pkl")