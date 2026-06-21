import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

# Define CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def predict_image(image_path, model_path='cifar10_mobilenetv2.keras', top_k=3):
    """Loads a saved model and predicts the class of an image.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the saved Keras model.
        top_k (int): Number of top predictions to display.
    """

    # 1. Load the saved model
    print(f"Loading model from '{model_path}'...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model. Did you run the training script? Details: {e}")
        return

    # 2. Load and preprocess the image
    print(f"Processing image '{image_path}'...")
    try:
        # Load image and resize to standard CIFAR-10 dimensions (32x32)
        # Note: The model handles internal resizing to 96x96 and preprocessing
        img = Image.open(image_path).convert('RGB')
        img = img.resize((32, 32))

        # Convert to numpy array and add batch dimension: shape → (1, 32, 32, 3)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return
    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")
        return

    # 3. Make the prediction
    predictions = model.predict(img_array, verbose=0)

    # 4. Interpret the results — show top-k predictions
    top_indices = np.argsort(predictions[0])[::-1][:top_k]

    print("\n" + "=" * 40)
    print(f"  Results for: {image_path}")
    print("=" * 40)
    for rank, idx in enumerate(top_indices, start=1):
        label = class_names[idx]
        confidence = predictions[0][idx] * 100
        bar = "█" * int(confidence / 5) + "░" * (20 - int(confidence / 5))
        marker = " ← TOP PICK" if rank == 1 else ""
        print(f"  #{rank}  {label:<12} {bar}  {confidence:5.2f}%{marker}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on an image using the trained CIFAR-10 MobileNetV2 model."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="OIP.jpg",
        help="Path to the input image (default: OIP.jpg)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="cifar10_mobilenetv2.keras",
        help="Path to the saved Keras model (default: cifar10_mobilenetv2.keras)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3)"
    )

    args = parser.parse_args()
    predict_image(
        image_path=args.image_path,
        model_path=args.model_path,
        top_k=args.top_k
    )