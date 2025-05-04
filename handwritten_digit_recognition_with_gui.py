import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

# Define the CNN model architecture identical to the trained one
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Function to load the pretrained model weights
# For demo purpose, will load and train model on MNIST here.
def train_and_load_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape((-1,28,28,1))
    x_test = x_test.reshape((-1,28,28,1))

    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Training model... (5 epochs, this might take a bit)")
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Trained model accuracy on MNIST test set: {acc*100:.2f}%")
    return model

# Main GUI class
class DigitRecognizerApp:
    def __init__(self, master, model):
        self.master = master
        self.master.title("Handwritten Digit Recognition")
        self.model = model

        # Canvas for drawing
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg='white', cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=4, pady=10, padx=10)

        # PIL image for drawing
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), color=255)  # white background
        self.draw = ImageDraw.Draw(self.image1)

        # Bind mouse events to canvas
        self.canvas.bind('<B1-Motion>', self.paint)  # mouse drag event
        self.canvas.bind('<ButtonRelease-1>', self.reset)

        # Variables for tracking paint
        self.last_x, self.last_y = None, None

        # Buttons
        self.predict_button = tk.Button(master, text="Predict", command=self.predict_digit, width=10)
        self.predict_button.grid(row=1, column=0, pady=10)

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas, width=10)
        self.clear_button.grid(row=1, column=1, pady=10)

        # Label to show prediction result
        self.result_label = tk.Label(master, text="Draw a digit and click Predict", font=("Helvetica", 16))
        self.result_label.grid(row=1, column=2, columnspan=2, padx=10)

    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # brush radius
        if self.last_x and self.last_y:
            # Draw on the canvas (black color)
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=r*2, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            # Draw on the PIL image as well (black color = 0)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=r*2)
        else:
            # First point, draw a circle (dot)
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
            self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        # Reset last coordinates when mouse button released
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=255)
        self.result_label.config(text="Draw a digit and click Predict")

    def preprocess_image(self):
        # Resize the PIL image to 28x28 pixels (MNIST standard)
        resized = self.image1.resize((28, 28), Image.LANCZOS)

        # Invert image: MNIST digits are white on black background; here it's black on white canvas
        inverted = ImageOps.invert(resized)

        # Normalize pixel values to [0,1]
        img_array = np.array(inverted).astype('float32') / 255.0

        # Reshape to fit model input (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array

    def predict_digit(self):
        img_array = self.preprocess_image()
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        self.result_label.config(text=f"Prediction: {digit} (Confidence: {confidence*100:.2f}%)")

def main():
    # Load or train model
    model = train_and_load_model()

    # Create the GUI
    root = tk.Tk()
    root.geometry("600x350")
    app = DigitRecognizerApp(root, model)
    root.mainloop()

if __name__ == '__main__':
    main()
