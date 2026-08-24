import json
import os
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes):
    """Convolutional Neural Network (CNN) for ECG Image Classification."""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),

        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Dense Classifier Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=32):
    """Build, train, and save the model."""
    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        )
    ]

    print("\nTraining CNN Model on ECG Images...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, "nn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f)
        print(f"Saved: {os.path.join(output_dir, 'nn_model.keras')}")

    return model, history


def predict_model(model, X_test):
    """Predict class indices."""
    probabilities = model.predict(X_test, verbose=0)
    return probabilities.argmax(axis=1)