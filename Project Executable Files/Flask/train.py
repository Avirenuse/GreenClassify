import os
import matplotlib.pyplot as plt
from keras.utils import load_img
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)

# Define paths
train_path = "Vegetable Images/train"
validation_path = "Vegetable Images/validation"
test_path = "Vegetable Images/test"

image_categories = os.listdir("Vegetable Images/train")


def plot_images(image_categories, train_path):
    """Plot first image from each vegetable category"""
    plt.figure(figsize=(12, 12))

    for i, cat in enumerate(image_categories):
        image_path = os.path.join(train_path, cat)
        images_in_folder = os.listdir(image_path)
        first_image_of_folder = images_in_folder[0]
        first_image_path = os.path.join(image_path, first_image_of_folder)
        img = load_img(first_image_path)
        img_arr = tf.keras.utils.img_to_array(img) / 255.0
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(cat)
        plt.axis("off")

    # Add an empty subplot for the 16th position
    plt.subplot(4, 4, 16)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("vegetable_samples.png")
    print("✓ Sample images plot saved as 'vegetable_samples.png'")
    plt.show()


def create_datasets_from_directory(
    train_path, validation_path, test_path, batch_size=32
):
    """Create datasets from directory structure"""
    print("\nLoading datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        label_mode="categorical",
        image_size=(256, 256),
        batch_size=batch_size,
        shuffle=True,
        seed=123,
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        validation_path,
        label_mode="categorical",
        image_size=(256, 256),
        batch_size=batch_size,
        shuffle=True,
        seed=123,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        label_mode="categorical",
        image_size=(256, 256),
        batch_size=batch_size,
        shuffle=True,
        seed=123,
    )

    print("✓ Datasets loaded successfully")
    return train_ds, validation_ds, test_ds


def label_encoding(train_ds):
    """Get class names and number of classes"""
    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(f"\nClass Name Mapping:")
    for i, class_name in enumerate(class_names):
        print(f"{i}: {class_name}")

    return class_names, num_classes


def resize_images(dataset):
    """Resize images to 150x150"""
    return dataset.map(lambda x, y: (tf.image.resize(x, (150, 150)), y))


def create_model(input_shape=(150, 150, 3), num_classes=15):
    """Create CNN model for vegetable classification"""
    print("\nBuilding CNN model...")

    model = Sequential()

    # First Conv Block
    model.add(
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape)
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Second Conv Block
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Third Conv Block
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Fourth Conv Block
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    print("✓ Model created successfully")
    return model


def train_model(model, train_ds, validation_ds, epochs=100):
    """Train the model"""
    print("\nTraining model...")

    early_stopping = keras.callbacks.EarlyStopping(patience=10)

    hist = model.fit(
        train_ds,
        epochs=epochs,
        verbose=1,
        validation_data=validation_ds,
        callbacks=early_stopping,
    )

    print("✓ Model training completed")
    return hist


def plot_training_history(hist):
    """Plot training and validation metrics"""
    print("\nPlotting training history...")

    h = hist.history
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(h["loss"], c="red", label="Training Loss")
    plt.plot(h["val_loss"], c="red", linestyle="--", label="Validation Loss")
    plt.plot(h["accuracy"], c="blue", label="Training Accuracy")
    plt.plot(h["val_accuracy"], c="blue", linestyle="--", label="Validation Accuracy")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss / Accuracy")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("✓ Training history plot saved as 'training_history.png'")
    plt.show()


def evaluate_model(model, test_ds):
    """Evaluate model on test dataset"""
    print("\nEvaluating model on test dataset...")

    loss, accuracy = model.evaluate(test_ds)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return loss, accuracy


def plot_predictions(model, train_ds, class_names):
    """Plot model predictions on training samples"""
    print("\nGenerating prediction plots...")

    plt.figure(figsize=(16, 16))

    for x, y in train_ds.take(1):
        pass  # Get a batch of data

    for i in range(32):
        plt.subplot(7, 5, i + 1)
        predicted_label = np.argmax(
            model.predict(np.reshape(x[i], (1, 150, 150, 3)), verbose=0)
        )
        plt.imshow(x[i].numpy().astype("uint8"))
        plt.title(
            f"Pred: {class_names[predicted_label]}\nActual: {class_names[np.argmax(y[i])]}",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 2},
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("predictions.png")
    print("✓ Predictions plot saved as 'predictions.png'")
    plt.show()


def main():
    """Main function to orchestrate the training pipeline"""
    print("=" * 50)
    print("VEGETABLE CLASSIFICATION MODEL TRAINING")
    print("=" * 50)

    # Step 1: Plot sample images
    print("\nStep 1: Displaying sample images from each category...")
    plot_images(image_categories, train_path)

    # Step 2: Create datasets
    print("\nStep 2: Creating datasets from directory...")
    train_ds, validation_ds, test_ds = create_datasets_from_directory(
        train_path, validation_path, test_path
    )

    # Step 3: Get class names
    print("\nStep 3: Encoding labels...")
    class_names, num_classes = label_encoding(train_ds)

    # Step 4: Resize images
    print("\nStep 4: Resizing images...")
    train_ds = resize_images(train_ds)
    validation_ds = resize_images(validation_ds)
    test_ds = resize_images(test_ds)
    print("✓ Images resized to 150x150")

    # Step 5: Create model
    print("\nStep 5: Creating model architecture...")
    model = create_model(num_classes=num_classes)
    print("\nModel Summary:")
    model.summary()

    # Step 6: Train model
    print("\nStep 6: Training model...")
    hist = train_model(model, train_ds, validation_ds, epochs=100)

    # Step 7: Plot training history
    print("\nStep 7: Visualizing training metrics...")
    plot_training_history(hist)

    # Step 8: Evaluate model
    print("\nStep 8: Evaluating on test set...")
    evaluate_model(model, test_ds)

    # Step 9: Plot predictions
    print("\nStep 9: Plotting sample predictions...")
    plot_predictions(model, train_ds, class_names)

    # Step 10: Save model
    print("\nStep 10: Saving model...")
    model.save("model.h5")
    print("✓ Model saved as 'model.h5'")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)

    return model, class_names


if __name__ == "__main__":
    model, class_names = main()
