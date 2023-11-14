# mojo_ai_variables.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the path to your dataset in Google Colab
dataset_path = "mojo_Dataset"

# Initialize empty lists to store text and labels
texts = []
labels = []

# Loop through each folder (examples, proposals, user, workshops)
folders = ["examples", "proposals", "user", "workshops"]

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)

    # Loop through each file in the folder
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Append the content to the texts list and the folder name to the labels list
            texts.append(content)
            labels.append(folder)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences for fixed input size
max_seq_length = max(map(len, sequences))
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

# Convert labels to numerical format
label_mapping = {label: index for index, label in enumerate(set(labels))}
numerical_labels = np.array([label_mapping[label] for label in labels])

# Save necessary variables to a file (mojo_ai_variables.npz)
np.savez('mojo_ai_variables.npz',
         tokenizer=tokenizer,
         label_mapping=label_mapping,
         max_seq_length=max_seq_length,
         padded_sequences=padded_sequences,
         numerical_labels=numerical_labels)
