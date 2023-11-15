# retrain_model_with_user_data.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load necessary variables from mojo_ai_variables.npz
loaded_variables = np.load('mojo_ai_variables.npz', allow_pickle=True)
tokenizer = loaded_variables['tokenizer'].item()
label_mapping = loaded_variables['label_mapping'].item()
max_seq_length = int(loaded_variables['max_seq_length'])
# Assuming you have padded_sequences and numerical_labels saved
padded_sequences = loaded_variables['padded_sequences']
numerical_labels = loaded_variables['numerical_labels']

# Define paths and directories
dataset_path = "mojo_Dataset"
user_folder_path = os.path.join(dataset_path, "user")
data_file_path = os.path.join(user_folder_path, "data")

# Load the model
loaded_model = tf.keras.models.load_model('mojo_ai_model.h5')

# Simulated user interaction
user_question = "How does Mojo handle interop with Python?"
user_answer = "Mojo uses a Python API for seamless interop."

# Store the user question and answer
user_data = [f"{user_question}\t{user_answer}\n"]

# Save the updated user data
with open(data_file_path, 'w', encoding='utf-8') as user_data_file:
    user_data_file.writelines(user_data)

# Tokenize the new user data
user_sequences = tokenizer.texts_to_sequences([user_question])
user_padded_sequences = pad_sequences(user_sequences, maxlen=max_seq_length, padding='post')

# Convert the label to numerical format
user_label = label_mapping["user"]

# Continue training with the new user data
loaded_model.fit(user_padded_sequences, np.array([user_label]), epochs=5, batch_size=1)
