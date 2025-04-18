import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model("C:/Users/menth/OneDrive/Documents/Code/dessert_recognition_model.keras")
img_height, img_width = 224, 224

def predict_dessert(image_path):
    from tensorflow.keras.preprocessing import image
    try:
        img = image.load_img(image_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        class_names = ['Kuih Bahulu', 'Kuih Ketayap', 'Kuih Seri Muka', 'Kuih Talam', 'Onde Onde']  # Matches your dataset
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f}) for {image_path}")
        plt.imshow(img)
        plt.title(f"{predicted_class} ({confidence:.2f})")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Directory containing test images
test_dir = "C:/Users/menth/Downloads/Test project"

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# Get all image files in the test directory
try:
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                   if os.path.isfile(os.path.join(test_dir, f)) and f.lower().endswith(image_extensions)]
    if not image_paths:
        print(f"No images found in {test_dir}")
    else:
        print(f"Testing {len(image_paths)} images from {test_dir}")
        for image_path in image_paths:
            predict_dessert(image_path)
except Exception as e:
    print(f"Error accessing {test_dir}: {e}")