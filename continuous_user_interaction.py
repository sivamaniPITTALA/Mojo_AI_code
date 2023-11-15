import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load necessary variables from mojo_ai_variables.npz
loaded_variables = np.load('mojo_ai_variables.npz', allow_pickle=True)
tokenizer = loaded_variables['tokenizer'].item()
label_mapping = loaded_variables['label_mapping'].item()
max_seq_length = int(loaded_variables['max_seq_length'])

# Load the model
loaded_model = tf.keras.models.load_model('mojo_ai_model_cnn_lstm.h5')

# Define paths and directories
dataset_path = "mojo_Dataset"
user_folder_path = os.path.join(dataset_path, "user")
data_file_path = os.path.join(user_folder_path, "data")

while True:
    # Get user input
    user_question = input("You: ")

    # Exit loop if user enters 'exit'
    if user_question.lower() == 'exit':
        break

    # Tokenize the user input
    user_sequences = tokenizer.texts_to_sequences([user_question])
    user_padded_sequences = pad_sequences(user_sequences, maxlen=max_seq_length, padding='post')

    # Get model prediction
    predictions = loaded_model.predict(user_padded_sequences)
    predicted_label = np.argmax(predictions)

    # Map predicted label to the corresponding category
    for label, index in label_mapping.items():
        if index == predicted_label:
            predicted_category = label

    # Provide response based on predicted category
    if predicted_category == 'examples':
        response = "This seems like an example."
    elif predicted_category == 'proposals':
        response = "It appears to be a proposal."
    elif predicted_category == 'user':
        response = "It looks like a user interaction."
    elif predicted_category == 'workshops':
        response = "Seems related to workshops."
    else:
        response = "Uncertain about the category."

    print(f"Bot: {response}")

    # Store user input and bot response
    user_data = [f"{user_question}\t{response}\n"]
    with open(data_file_path, 'a', encoding='utf-8') as user_data_file:
        user_data_file.writelines(user_data)
