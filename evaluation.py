def evaluate_model(model, features, labels):
    """Evaluate the global model on the combined dataset."""
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    loss, accuracy = model.evaluate(features, labels, batch_size=32)
    return loss, accuracy
