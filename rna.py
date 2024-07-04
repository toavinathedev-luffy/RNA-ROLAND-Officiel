# try:
#     import _bootlocale
# except ImportError:
#     pass

# Votre code existant ici


import random
import math
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Henon function parameters
a = 1.4
b = 0.3

# Generate Henon data
def henon(x, y):
    return 1 - a * x * x + y, b * x

# Initialize neural network weights
def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
    weights_hidden_output = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
    return weights_input_hidden, weights_hidden_output

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Feedforward function
def feedforward(inputs, weights_input_hidden, weights_hidden_output):
    hidden_layer_activation = [0] * len(weights_input_hidden)
    for i in range(len(weights_input_hidden)):
        hidden_layer_activation[i] = sigmoid(sum(inputs[j] * weights_input_hidden[i][j] for j in range(len(inputs))))
    
    output_layer_activation = [0] * len(weights_hidden_output)
    for i in range(len(weights_hidden_output)):
        output_layer_activation[i] = sigmoid(sum(hidden_layer_activation[j] * weights_hidden_output[i][j] for j in range(len(hidden_layer_activation))))
    
    return hidden_layer_activation, output_layer_activation

# Backpropagation function
def backpropagate(inputs, expected_output, hidden_layer_activation, output_layer_activation, weights_input_hidden, weights_hidden_output, learning_rate):
    output_error = [expected_output[i] - output_layer_activation[i] for i in range(len(expected_output))]
    output_delta = [output_error[i] * sigmoid_derivative(output_layer_activation[i]) for i in range(len(output_error))]
    
    hidden_error = [0] * len(hidden_layer_activation)
    for i in range(len(hidden_layer_activation)):
        hidden_error[i] = sum(output_delta[j] * weights_hidden_output[j][i] for j in range(len(output_delta)))
    
    hidden_delta = [hidden_error[i] * sigmoid_derivative(hidden_layer_activation[i]) for i in range(len(hidden_error))]
    
    # Update weights
    for i in range(len(weights_hidden_output)):
        for j in range(len(weights_hidden_output[i])):
            weights_hidden_output[i][j] += learning_rate * output_delta[i] * hidden_layer_activation[j]
    
    for i in range(len(weights_input_hidden)):
        for j in range(len(weights_input_hidden[i])):
            weights_input_hidden[i][j] += learning_rate * hidden_delta[i] * inputs[j]

# Predict for multiple steps
def predict(steps, x, y, weights_input_hidden, weights_hidden_output):
    predictions = []
    for _ in range(steps):
        inputs = [x, y]
        _, output = feedforward(inputs, weights_input_hidden, weights_hidden_output)
        predictions.append(output)
        x, y = output
    return predictions

# Tkinter interface
class HenonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Série de Henon et Réseaux de Neurones")

        # Parameters
        self.n_points = 500
        self.x0, self.y0 = 0, 0  # Initial conditions
        self.learning_rate = 0.1
        self.epochs = 1000
        self.prediction_steps = 3

        # Initialize weights
        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 2
        self.weights_input_hidden, self.weights_hidden_output = initialize_weights(self.input_size, self.hidden_size, self.output_size)

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Buttons and Entries

        self.generate_button = ttk.Button(self.root, text="Générer les données", command=self.generate_data)
        self.generate_button.grid(row=0, column=0, padx=10, pady=10)

        self.train_button = ttk.Button(self.root, text="Entraîner le réseau", command=self.train_model)
        self.train_button.grid(row=0, column=1, padx=10, pady=10)

        self.predict_button = ttk.Button(self.root, text="Prédire")
        self.predict_button.grid(row=0, column=2, padx=10, pady=10)

        self.predict_1_step_button = ttk.Button(self.root, text="Prédire 1 pas", command=lambda: self.predict(1))
        self.predict_1_step_button.grid(row=0, column=3, padx=10, pady=10)

        self.predict_10_steps_button = ttk.Button(self.root, text="Prédire 10 pas", command=lambda: self.predict(10))
        self.predict_10_steps_button.grid(row=0, column=4, padx=10, pady=10)

        self.predict_20_steps_button = ttk.Button(self.root, text="Prédire 20 pas", command=lambda: self.predict(20))
        self.predict_20_steps_button.grid(row=0, column=5, padx=10, pady=10)

        # Plot area
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().grid(row=1, column=2, columnspan=7)

        # Prediction value display
        self.prediction_label = ttk.Label(self.root, text="Prédictions :")
        self.prediction_label.grid(row=1, column=2)
        self.prediction_value = ttk.Label(self.root, text="")
        self.prediction_value.grid(row=1, column=2, columnspan=6)

    def generate_data(self):
        self.training_data = []
        x, y = self.x0, self.y0
        for _ in range(self.n_points):
            x, y = henon(x, y)
            self.training_data.append((x, y))

        self.ax.clear()
        self.ax.scatter([d[0] for d in self.training_data], [d[1] for d in self.training_data], color='green', marker='o', s=2)
        self.canvas.draw()
        # self.ax.plot([d[0] for d in self.training_data], [d[1] for d in self.training_data])
      


    def train_model(self):
        for _ in range(self.epochs):
            for x, y in self.training_data:
                inputs = [x, y]
                expected_output = henon(x, y)
                hidden_layer_activation, output_layer_activation = feedforward(inputs, self.weights_input_hidden, self.weights_hidden_output)
                backpropagate(inputs, expected_output, hidden_layer_activation, output_layer_activation, self.weights_input_hidden, self.weights_hidden_output, self.learning_rate)
        # Display training completion message
        message = "Entraînement terminé !"
        self.prediction_value.config(text=message)
        tk.messagebox.showinfo("Information", message)  # Display training completion popup

    def predict(self, steps):
        x, y = self.x0, self.y0
        predictions = predict(steps, x, y, self.weights_input_hidden, self.weights_hidden_output)
        self.ax.clear()
        self.ax.scatter([d[0] for d in self.training_data], [d[1] for d in self.training_data], color='green', marker='o', s=2)
        # self.ax.plot([d[0] for d in self.training_data], [d[1] for d in self.training_data])
        self.ax.scatter([p[0] for p in predictions], [p[1] for p in predictions], color='red', marker='o', s=2)
        # self.ax.plot([p[0] for p in predictions], [p[1] for p in predictions], 'r')
        self.canvas.draw()

        # Update prediction value display
        prediction_text = "Prédiction à " + str(steps) + " pas :\n"
        for i, prediction in enumerate(predictions):
            prediction_text += f"  - Pas {i+1}: ({prediction[0]:.4f}, {prediction[1]:.4f})\n"
        self.prediction_value.config(text=prediction_text)  

        # Display prediction completion message
        message = "Prédiction terminée !"
        tk.messagebox.showinfo("Information", message)  # Display prediction completion popup

# Tkinter interface
if __name__ == "__main__":
    root = tk.Tk()
    app = HenonApp(root)
    root.mainloop()
