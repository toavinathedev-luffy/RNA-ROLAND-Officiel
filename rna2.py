import random
import math
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
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
    return (
        [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)],
        [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
    )

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error (MSE) function
def mean_squared_error(predictions, targets):
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

# Feedforward function
def feedforward(inputs, weights_input_hidden, weights_hidden_output):
    hidden_layer_activation = [sigmoid(sum(inputs[j] * weights_input_hidden[i][j] for j in range(len(inputs)))) for i in range(len(weights_input_hidden))]
    output_layer_activation = [sigmoid(sum(hidden_layer_activation[j] * weights_hidden_output[i][j] for j in range(len(hidden_layer_activation)))) for i in range(len(weights_hidden_output))]
    return hidden_layer_activation, output_layer_activation

# Backpropagation function
def backpropagate(inputs, expected_output, hidden_layer_activation, output_layer_activation, weights_input_hidden, weights_hidden_output, learning_rate):
    output_error = [expected_output[i] - output_layer_activation[i] for i in range(len(expected_output))]
    output_delta = [output_error[i] * sigmoid_derivative(output_layer_activation[i]) for i in range(len(output_error))]
    hidden_error = [sum(output_delta[j] * weights_hidden_output[j][i] for j in range(len(output_delta))) for i in range(len(hidden_layer_activation))]
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

        # Training data and error history
        self.training_data = []
        self.error_history = []

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Buttons
        buttons = [
            ("Générer les données", self.generate_data),
            ("Entraîner le réseau", self.train_model),
            ("Prédire 1 pas", lambda: self.predict(1)),
            ("Prédire 10 pas", lambda: self.predict(10)),
            ("Prédire 20 pas", lambda: self.predict(20)),
        ]
        
        for idx, (text, command) in enumerate(buttons):
            button = ttk.Button(self.root, text=text, command=command)
            button.grid(row=0, column=idx, padx=10, pady=10)

        # Plot area for Henon data
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=5)

        # Prediction value display
        self.prediction_label = ttk.Label(self.root, text="Prédictions :")
        self.prediction_label.grid(row=2, column=0, sticky='w')
        self.prediction_value = ttk.Label(self.root, text="")
        self.prediction_value.grid(row=2, column=1, columnspan=4, sticky='w')

        # Plot area for error curve
        self.error_figure, self.error_ax = plt.subplots(figsize=(8, 2))
        self.error_canvas = FigureCanvasTkAgg(self.error_figure, self.root)
        self.error_canvas.get_tk_widget().grid(row=3, column=0, columnspan=5)

    def generate_data(self):
        self.training_data = []
        x, y = self.x0, self.y0
        for _ in range(self.n_points):
            x, y = henon(x, y)
            self.training_data.append((x, y))

        self.ax.clear()
        self.ax.scatter(*zip(*self.training_data), color='green', marker='o', s=2)
        self.canvas.draw()

    def train_model(self):
        self.error_history = []
        for epoch in range(self.epochs):
            epoch_error = 0
            for x, y in self.training_data:
                inputs = [x, y]
                expected_output = henon(x, y)
                hidden_layer_activation, output_layer_activation = feedforward(inputs, self.weights_input_hidden, self.weights_hidden_output)
                backpropagate(inputs, expected_output, hidden_layer_activation, output_layer_activation, self.weights_input_hidden, self.weights_hidden_output, self.learning_rate)
                epoch_error += mean_squared_error(output_layer_activation, expected_output)
            
            self.error_history.append(epoch_error / len(self.training_data))
        
        self.update_error_plot()
        message = "Entraînement terminé !"
        self.prediction_value.config(text=message)
        messagebox.showinfo("Information", message)

    def predict(self, steps):
        x, y = self.x0, self.y0
        predictions = predict(steps, x, y, self.weights_input_hidden, self.weights_hidden_output)
        self.ax.clear()
        self.ax.scatter(*zip(*self.training_data), color='green', marker='o', s=2)
        self.ax.scatter(*zip(*predictions), color='red', marker='o', s=2)
        self.canvas.draw()

        if steps == 1:
            message = f"Prédiction à 1 pas : ({predictions[0][0]:.4f}, {predictions[0][1]:.4f})"
            self.prediction_value.config(text=message)
            messagebox.showinfo("Information", message)
        else:
            self.show_prediction_curve(predictions)

    def show_prediction_curve(self, predictions):
        prediction_window = Toplevel(self.root)
        prediction_window.title("Prédictions")
        fig, ax = plt.subplots()
        canvas = FigureCanvasTkAgg(fig, prediction_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Plotting the prediction data for X values only
        real_x = [d[0] for d in self.training_data[-len(predictions):]]
        pred_x = [p[0] for p in predictions]

        ax.plot(real_x, label='Valeurs réelles X', linestyle='-', marker='o', color='blue')
        ax.plot(pred_x, label='Valeurs prédites X', linestyle='--', marker='x', color='red')

        ax.legend()
        ax.set_title(f"Prédictions à {len(predictions)} pas")
        ax.set_xlabel("Étapes de prédiction")
        ax.set_ylabel("Valeurs de X")
        canvas.draw()

    def update_error_plot(self):
        self.error_ax.clear()
        epochs = range(1, self.epochs + 1)
        self.error_ax.plot(epochs, self.error_history, label='Erreur quadratique moyenne')
        self.error_ax.set_title('Evolution de l\'erreur pendant l\'entraînement')
        self.error_ax.set_xlabel('Époques')
        self.error_ax.set_ylabel('Erreur')
        self.error_ax.legend()
        self.error_canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = HenonApp(root)
    root.mainloop()
