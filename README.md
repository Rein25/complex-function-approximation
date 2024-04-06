
# Sine Wave Approach Project

## Overview

This project demonstrates the use of neural networks to approximate mathematical functions, focusing on the ability to closely model sinusoidal functions of varying complexities. It leverages PyTorch, a leading deep learning library, to construct and train the neural network models. The versatility of the model is showcased by enabling user input for defining the target mathematical function, including high-frequency sine waves and other complex functions.

## Features

- **Neural Network Modeling**: Uses PyTorch for constructing and training neural networks.
- **Dynamic Function Input**: Allows users to input arbitrary mathematical functions, which the neural network will then learn to approximate.
- **Visualization**: Includes real-time plotting of the target function and the neural network's approximation, providing immediate visual feedback on the model's performance.
- **Support for Complex Functions**: Capable of handling a wide range of functions, including trigonometric, polynomial, and hyperbolic functions.

## Contents

- `neural_network_model.py`: The main script which sets up the neural network, trains it on user-specified functions, and visualizes the results.
- `requirements.txt`: A file listing all the necessary Python libraries required to run the project.

## Getting Started

### Prerequisites

Ensure you have Python 3.6 or later installed on your system. This project also requires the following Python libraries:

- torch
- numpy
- matplotlib
- sympy

### Installation

1. Clone the repository or download the source code.
2. Install the required Python libraries by running the following command in your terminal:

   ```
   pip install -r requirements.txt
   ```

### Usage

To run the project, navigate to the project directory in your terminal and execute the following command:

```
python neural_network_model.py
```

Upon running the script, you will be prompted to input a mathematical function. Enter a function using `x` as the variable (e.g., `sin(x) + x**2`). The program will then begin training a neural network to approximate this function and visualize the results in real-time.

## Contributing

Contributions to the project are welcome! Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Thanks to PyTorch for providing an intuitive framework for deep learning research and development.
- Appreciation for the contributors who have offered valuable insights and improvements.
