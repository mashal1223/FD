# F:\FD\main.py

import os
from data.preprocess import preprocess_dataset
from clients.client_training import train_client_model

# Path to the dataset
DATA_PATH = os.path.join(os.getcwd(), "healthcare_dataset.csv")

# Preprocess dataset
client1_data, client2_data, encoders = preprocess_dataset(DATA_PATH)

# Train models for Client 1 and Client 2
print("Training Client 1 model...")
client1_model, client1_accuracy = train_client_model(client1_data)
print(f"Client 1 Model Accuracy: {client1_accuracy:.2f}")

print("Training Client 2 model...")
client2_model, client2_accuracy = train_client_model(client2_data)
print(f"Client 2 Model Accuracy: {client2_accuracy:.2f}")

# Federated Averaging (Example)
# In real federated learning, this would involve secure aggregation.
print("Combining models in the main server...")
avg_accuracy = (client1_accuracy + client2_accuracy) / 2
print(f"Federated Average Accuracy: {avg_accuracy:.2f}")
