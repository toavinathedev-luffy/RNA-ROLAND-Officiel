import random
import math
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class HenonMap:
    def __init__(self, a=1.4, b=0.3):
        self.a = a
        self.b = b

    def generate_data(self, n_points, x0=0, y0=0):
        x, y = x0, y0
        data = []
        for _ in range(n_points):
            x, y = self.henon(x, y)
            data.append((x, y))
        return data

    def henon(self, x, y):
        return 1 - self.a * x * x + y, self.b * x

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def mean_squared_error(self, predictions, targets):
        return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

    def feedforward(self, inputs):
        hidden_layer_activation = [self.sigmoid(sum(inputs[j] * self.weights_input_hidden[i][j] for j in range(len(inputs)))) for i in range(len(self.weights_input_hidden))]
        output_layer_activation = [self.sigmoid(sum(hidden_layer_activation[j] * self.weights_hidden_output[i][j] for j in range(len(hidden_layer_activation)))) for i in range(len(self.weights_hidden_output))]
        return hidden_layer_activation, output_layer_activation

    def backpropagate(self, inputs, expected_output, hidden_layer_activation, output_layer_activation):
        output_error = [expected_output[i] - output_layer_activation[i] for i in range(len(expected_output))]
        output_delta = [output_error[i] * self.sigmoid_derivative(output_layer_activation[i]) for i in range(len(output_error))]
        hidden_error = [sum(output_delta[j] * self.weights_hidden_output[j][i] for j in range(len(output_delta))) for i in range(len(hidden_layer_activation))]
        hidden_delta = [hidden_error[i] * self.sigmoid_derivative(hidden_layer_activation[i]) for i in range(len(hidden_error))]

        # Update weights
        for i in range(len(self.weights_hidden_output)):
            for j in range(len(self.weights_hidden_output[i])):
                self.weights_hidden_output[i][j] += self.learning_rate * output_delta[i] * hidden_layer_activation[j]

        for i in range(len(self.weights_input_hidden)):
            for j in range(len(self.weights_input_hidden[i])):
                self.weights_input_hidden[i][j] += self.learning_rate * hidden_delta[i] * inputs[j]

    def train(self, training_data, epochs):
        error_history = []
        for epoch in range(epochs):
            epoch_error = 0
            for x, y in training_data:
                inputs = [x, y]
                expected_output = HenonMap().henon(x, y)
                hidden_layer_activation, output_layer_activation = self.feedforward(inputs)
                self.backpropagate(inputs, expected_output, hidden_layer_activation, output_layer_activation)
                epoch_error += self.mean_squared_error(output_layer_activation, expected_output)
            error_history.append(epoch_error / len(training_data))
        return error_history

    def predict(self, steps, x, y):
        predictions = []
        for _ in range(steps):
            inputs = [x, y]
            _, output = self.feedforward(inputs)
            predictions.append(output)
            x, y = output
        return predictions

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

        # Initialize Henon map and neural network
        self.henon_map = HenonMap()
        self.neural_network = NeuralNetwork(input_size=2, hidden_size=2, output_size=2, learning_rate=self.learning_rate)

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
            ("Quitter", self.quit_app),
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
        self.training_data = self.henon_map.generate_data(self.n_points, self.x0, self.y0)
        self.ax.clear()
        self.ax.scatter(*zip(*self.training_data), color='green', marker='o', s=2)
        self.canvas.draw()

    def train_model(self):
        self.error_history = self.neural_network.train(self.training_data, self.epochs)
        self.update_error_plot()
        message = "Entraînement terminé !"
        self.prediction_value.config(text=message)
        messagebox.showinfo("Information", message)

    def predict(self, steps):
        predictions = self.neural_network.predict(steps, self.x0, self.y0)
        self.ax.clear()
        self.ax.scatter(*zip(*self.training_data), color='green', marker='o', s=2)
        self.ax.scatter(*zip(*predictions), color='red', marker='o', s=2)
        self.canvas.draw()

        if steps == 1:
            message = f"Prédiction à 1 pas : ({predictions[0][0]:.4f}, {predictions[0][1]:.4f})"
            self.prediction_value.config(text=message)
            messagebox.showinfo("Information", message)
            print(message)  # Affiche dans la console
        else:
            self.show_prediction_curve(predictions)

    def show_prediction_curve(self, predictions):
        # Préparation du message à afficher dans la console
        prediction_message = f"Prédictions à {len(predictions)} pas :\n"
        for i, pred in enumerate(predictions):
            prediction_message += f"Étape {i+1}: ({pred[0]:.4f}, {pred[1]:.4f})\n"
        
        # Affichage du message dans la console
        print(prediction_message)

        # Création de la fenêtre de prédiction
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

        # Ajout d'un callback pour afficher le message dans la console après la fermeture de la fenêtre
        def on_close():
            prediction_window.destroy()
            messagebox.showinfo("Prédictions", prediction_message)

        prediction_window.protocol("WM_DELETE_WINDOW", on_close)


    def update_error_plot(self):
        self.error_ax.clear()
        epochs = range(1, self.epochs + 1)
        self.error_ax.plot(epochs, self.error_history, label='Erreur quadratique moyenne')
        self.error_ax.set_title('Evolution de l\'erreur pendant l\'entraînement')
        self.error_ax.set_xlabel('Époques')
        self.error_ax.set_ylabel('Erreur')
        self.error_ax.legend()
        self.error_canvas.draw()

    def quit_app(self):
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HenonApp(root)
    root.mainloop()
