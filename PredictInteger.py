import pandas as pd
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model

class ImagePredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict_image(self, image_path):
        try:
            image = Image.open(image_path)
            img = image.resize((28, 28), Image.LANCZOS)
            img = img.convert('L')
            image_array = np.array(img)
            df = pd.DataFrame(image_array)

            if df.values.max() == 255:
                X = df / 255.0
                X = X.values.reshape(-1, 28, 28, 1)

                y_pred = self.model.predict(X)
                predicted_class = np.argmax(y_pred, axis=1)

                return predicted_class[0]

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

def main():
    root = tk.Tk()
    root.withdraw()
    
    model_path = 'mnist_model.h5'  
    predictor = ImagePredictor(model_path)
    
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])

    if file_path:
        predicted_class = predictor.predict_image(file_path)
        if predicted_class is not None:
            print("Predicted Class:", predicted_class)
    else:
        print("No file selected")

if __name__ == "__main__":
    main()
