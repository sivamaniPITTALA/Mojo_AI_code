import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

# Load necessary variables from mojo_ai_variables.npz
loaded_variables = np.load('mojo_ai_variables.npz', allow_pickle=True)
tokenizer = loaded_variables['tokenizer'].item()
label_mapping = loaded_variables['label_mapping'].item()
max_seq_length = int(loaded_variables['max_seq_length'])
# Assuming you have padded_sequences and numerical_labels saved
padded_sequences = loaded_variables['padded_sequences']
numerical_labels = loaded_variables['numerical_labels']

# Load the saved model
model_path = 'mojo_ai_model_cnn_lstm.h5'
loaded_model = load_model(model_path)

# New data for prediction (replace this with your actual data)
new_texts = ["what is mojo", "why mojo is faster", "write hello word program in mojo"]

# Tokenize and pad the new text data
new_sequences = tokenizer.texts_to_sequences(new_texts)
padded_new_sequences = pad_sequences(new_sequences, maxlen=max_seq_length, padding='post')

# Perform predictions on the new data
predictions = loaded_model.predict(padded_new_sequences)

# Convert predictions to labels using label mapping
predicted_labels = [list(label_mapping.keys())[np.argmax(pred)] for pred in predictions]

# Display predicted labels for the new data
for i, text in enumerate(new_texts):
    print(f"Text: {text} - Predicted Label: {predicted_labels[i]}")
