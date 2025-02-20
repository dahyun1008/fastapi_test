import numpy as np
import json

class AndModel:
    def __init__(self):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self, epochs=20, learning_rate=0.1):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 0, 0, 1])

        for epoch in range(epochs):
            for i in range(len(inputs)):
                total_input = np.dot(inputs[i], self.weights) + self.bias
                prediction = self.step_function(total_input)
                error = outputs[i] - prediction

                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error

        print("Training complete.")
        self.save_weights_to_json()

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)

    def save_weights_to_json(self, file_path="and_model_weights.json"):
        model_data = {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist()
        }
        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=4)
        print(f"Model weights saved to {file_path}")

    def load_weights_from_json(self, file_path="and_model_weights.json"):
        with open(file_path, "r") as f:
            model_data = json.load(f)
        
        self.weights = np.array(model_data["weights"])
        self.bias = np.array(model_data["bias"])
        print(f"Model weights loaded from {file_path}")
