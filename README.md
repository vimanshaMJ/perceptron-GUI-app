# Advanced Perceptron GUI Application

A comprehensive desktop application for training and visualizing perceptron classifiers with an intuitive graphical interface.

## Features

- **Interactive Data Generation**: Create linearly separable, blob, or XOR datasets
- **Custom Point Addition**: Click on the plot to add custom data points
- **Real-time Training Visualization**: Watch the decision boundary update during training
- **Comprehensive Statistics**: View detailed model and dataset statistics
- **Training History**: Visualize error and accuracy curves
- **Export Functionality**: Save models, data, and plots
- **Parameter Tuning**: Adjust learning rate and epochs with interactive controls

## Installation

1. Clone the repository:
git clone <repository-url>
cd perceptron_gui_app
  
2. Create a virtual environment:
python -m venv venv
venv\Scripts\activate


3. Install dependencies:
pip install -r requirements.txt


## Usage

Run the application:
python main.py


### How to Use

1. **Generate Data**: Choose from predefined datasets or load your own CSV
2. **Add Custom Points**: Select class and click on the plot to add points
3. **Set Parameters**: Adjust learning rate and number of epochs
4. **Train Model**: Click "Train Perceptron" to start training
5. **View Results**: Check statistics and training history tabs
6. **Export**: Save your model, data, or plots

## Project Structure

```
perceptron_gui_app/
├── src/
│ ├── models/perceptron.py 
│ ├── gui/main_window.py 
│ └── data/data_generator.py 
├── assets/ 
├── main.py 
├── requirements.txt
└── README.md 
```


## License

MIT License
