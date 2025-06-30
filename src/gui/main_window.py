import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkinter, NavigationToolbar2Tk
import threading

from ..models.perceptron import AdvancedPerceptron
from ..data.data_generator import DataGenerator

class PerceptronGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Perceptron Classifier")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.perceptron = None
        self.X_data = None
        self.y_data = None
        self.custom_points = {'X': [], 'y': []}
        self.is_training = False
        self.current_class = 0
        
        self.setup_gui()
        self.generate_default_data()
    
    def setup_gui(self):
        # Create main frames
        self.create_control_panel()
        self.create_visualization_panel()
        self.create_results_panel()
    
    def create_control_panel(self):
        # Control Panel Frame
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Data Generation Section
        data_frame = ttk.LabelFrame(control_frame, text="Data Generation", padding="5")
        data_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Button(data_frame, text="Linear Separable", 
                  command=self.generate_linear_data).grid(row=0, column=0, padx=2)
        ttk.Button(data_frame, text="Blobs", 
                  command=self.generate_blob_data).grid(row=0, column=1, padx=2)
        ttk.Button(data_frame, text="XOR (Non-separable)", 
                  command=self.generate_xor_data).grid(row=0, column=2, padx=2)
        ttk.Button(data_frame, text="Load CSV", 
                  command=self.load_csv_data).grid(row=0, column=3, padx=2)
        
        # Custom Point Addition
        custom_frame = ttk.LabelFrame(control_frame, text="Add Custom Points", padding="5")
        custom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(custom_frame, text="Click on plot to add points").grid(row=0, column=0, columnspan=2)
        
        self.class_var = tk.IntVar(value=0)
        ttk.Radiobutton(custom_frame, text="Class 0", variable=self.class_var, 
                       value=0).grid(row=1, column=0)
        ttk.Radiobutton(custom_frame, text="Class 1", variable=self.class_var, 
                       value=1).grid(row=1, column=1)
        
        ttk.Button(custom_frame, text="Clear Points", 
                  command=self.clear_custom_points).grid(row=1, column=2, padx=5)
        
        # Training Parameters
        param_frame = ttk.LabelFrame(control_frame, text="Training Parameters", padding="5")
        param_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(param_frame, text="Learning Rate:").grid(row=0, column=0, sticky="w")
        self.lr_var = tk.DoubleVar(value=0.01)
        lr_scale = ttk.Scale(param_frame, from_=0.001, to=0.1, variable=self.lr_var, 
                            orient="horizontal", length=150)
        lr_scale.grid(row=0, column=1, sticky="ew")
        self.lr_label = ttk.Label(param_frame, text="0.01")
        self.lr_label.grid(row=0, column=2)
        lr_scale.configure(command=self.update_lr_label)
        
        ttk.Label(param_frame, text="Epochs:").grid(row=1, column=0, sticky="w")
        self.epochs_var = tk.IntVar(value=100)
        ttk.Spinbox(param_frame, from_=10, to=1000, textvariable=self.epochs_var, 
                   width=10).grid(row=1, column=1, sticky="w")
        
        # Training Controls
        train_frame = ttk.LabelFrame(control_frame, text="Training", padding="5")
        train_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.train_button = ttk.Button(train_frame, text="Train Perceptron", 
                                      command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(train_frame, text="Stop Training", 
                                     command=self.stop_training, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)
        
        ttk.Button(train_frame, text="Reset Model", 
                  command=self.reset_model).grid(row=0, column=2, padx=5)
        
        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(train_frame, variable=self.progress_var, 
                                           maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Status Label
        self.status_label = ttk.Label(train_frame, text="Ready to train")
        self.status_label.grid(row=2, column=0, columnspan=3)
    
    def create_visualization_panel(self):
        # Visualization Panel
        viz_frame = ttk.LabelFrame(self.root, text="Visualization", padding="5")
        viz_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkinter(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, viz_frame)
        toolbar.update()
        
        # Bind click event for custom point addition
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
    
    def create_results_panel(self):
        # Results Panel
        results_frame = ttk.LabelFrame(self.root, text="Results & Statistics", padding="5")
        results_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(results_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Statistics Tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        
        self.stats_text = tk.Text(stats_frame, height=8, width=50)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Training History Tab
        history_frame = ttk.Frame(notebook)
        notebook.add(history_frame, text="Training History")
        
        # Create training history plot
        self.history_fig = Figure(figsize=(6, 4), dpi=100)
        self.history_canvas = FigureCanvasTkinter(self.history_fig, history_frame)
        self.history_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Export Tab
        export_frame = ttk.Frame(notebook)
        notebook.add(export_frame, text="Export")
        
        ttk.Button(export_frame, text="Export Model", 
                  command=self.export_model).pack(pady=5)
        ttk.Button(export_frame, text="Export Data", 
                  command=self.export_data).pack(pady=5)
        ttk.Button(export_frame, text="Export Plot", 
                  command=self.export_plot).pack(pady=5)
    
    def update_lr_label(self, value):
        self.lr_label.config(text=f"{float(value):.3f}")
    
    def generate_default_data(self):
        self.generate_linear_data()
    
    def generate_linear_data(self):
        self.X_data, self.y_data = DataGenerator.generate_linearly_separable()
        self.custom_points = {'X': [], 'y': []}
        self.plot_data()
        self.update_stats()
    
    def generate_blob_data(self):
        self.X_data, self.y_data = DataGenerator.generate_blobs()
        self.custom_points = {'X': [], 'y': []}
        self.plot_data()
        self.update_stats()
    
    def generate_xor_data(self):
        self.X_data, self.y_data = DataGenerator.generate_xor_data()
        self.custom_points = {'X': [], 'y': []}
        self.plot_data()
        self.update_stats()
    
    def load_csv_data(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.read_csv(file_path)
                if df.shape[1] < 3:
                    messagebox.showerror("Error", "CSV must have at least 3 columns (x1, x2, target)")
                    return
                
                self.X_data = df.iloc[:, :2].values
                self.y_data = df.iloc[:, 2].values
                self.custom_points = {'X': [], 'y': []}
                self.plot_data()
                self.update_stats()
                messagebox.showinfo("Success", f"Loaded {len(self.X_data)} data points")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def on_plot_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        class_label = self.class_var.get()
        self.custom_points['X'].append([x, y])
        self.custom_points['y'].append(class_label)
        
        self.plot_data()
    
    def clear_custom_points(self):
        self.custom_points = {'X': [], 'y': []}
        self.plot_data()
    
    def plot_data(self):
        self.ax.clear()
        
        if self.X_data is not None:
            # Plot original data
            class_0_mask = self.y_data == 0
            class_1_mask = self.y_data == 1
            
            self.ax.scatter(self.X_data[class_0_mask, 0], self.X_data[class_0_mask, 1], 
                           c='red', marker='o', s=50, alpha=0.7, label='Class 0')
            self.ax.scatter(self.X_data[class_1_mask, 0], self.X_data[class_1_mask, 1], 
                           c='blue', marker='^', s=50, alpha=0.7, label='Class 1')
        
        # Plot custom points
        if self.custom_points['X']:
            custom_X = np.array(self.custom_points['X'])
            custom_y = np.array(self.custom_points['y'])
            
            custom_0_mask = custom_y == 0
            custom_1_mask = custom_y == 1
            
            if np.any(custom_0_mask):
                self.ax.scatter(custom_X[custom_0_mask, 0], custom_X[custom_0_mask, 1], 
                               c='darkred', marker='s', s=80, label='Custom Class 0')
            if np.any(custom_1_mask):
                self.ax.scatter(custom_X[custom_1_mask, 0], custom_X[custom_1_mask, 1], 
                               c='darkblue', marker='D', s=80, label='Custom Class 1')
        
        # Plot decision boundary if model is trained
        if self.perceptron and self.perceptron.is_trained:
            x_boundary, y_boundary = self.perceptron.get_decision_boundary()
            if x_boundary and y_boundary:
                self.ax.plot(x_boundary, y_boundary, 'k-', linewidth=2, 
                            label='Decision Boundary')
        
        self.ax.set_xlabel('Feature 1')
        self.ax.set_ylabel('Feature 2')
        self.ax.set_title('Perceptron Classification')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def get_combined_data(self):
        if self.X_data is None:
            if not self.custom_points['X']:
                return None, None
            return np.array(self.custom_points['X']), np.array(self.custom_points['y'])
        
        if not self.custom_points['X']:
            return self.X_data, self.y_data
        
        # Combine original and custom data
        combined_X = np.vstack([self.X_data, np.array(self.custom_points['X'])])
        combined_y = np.hstack([self.y_data, np.array(self.custom_points['y'])])
        
        return combined_X, combined_y
    
    def start_training(self):
        X, y = self.get_combined_data()
        if X is None or len(X) < 2:
            messagebox.showerror("Error", "Need at least 2 data points to train")
            return
        
        self.is_training = True
        self.train_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Create new perceptron
        self.perceptron = AdvancedPerceptron(
            num_features=2, 
            learning_rate=self.lr_var.get()
        )
        
        # Start training in separate thread
        self.training_thread = threading.Thread(
            target=self.train_perceptron, 
            args=(X, y, self.epochs_var.get())
        )
        self.training_thread.start()
    
    def train_perceptron(self, X, y, epochs):
        def training_callback(epoch, error, accuracy):
            if not self.is_training:
                return
            
            # Update GUI in main thread
            self.root.after(0, self.update_training_progress, epoch, error, accuracy, epochs)
        
        try:
            self.perceptron.fit(X, y, epochs=epochs, callback=training_callback)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        finally:
            self.root.after(0, self.training_finished)
    
    def update_training_progress(self, epoch, error, accuracy, total_epochs):
        progress = (epoch / total_epochs) * 100
        self.progress_var.set(progress)
        self.status_label.config(text=f"Epoch {epoch}/{total_epochs} - Error: {error} - Accuracy: {accuracy:.3f}")
        
        # Update plot every 10 epochs or on last epoch
        if epoch % 10 == 0 or epoch == total_epochs:
            self.plot_data()
            self.plot_training_history()
    
    def training_finished(self):
        self.is_training = False
        self.train_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_var.set(100)
        self.status_label.config(text="Training completed!")
        
        self.plot_data()
        self.plot_training_history()
        self.update_stats()
    
    def stop_training(self):
        self.is_training = False
        self.status_label.config(text="Training stopped by user")
    
    def reset_model(self):
        self.perceptron = None
        self.progress_var.set(0)
        self.status_label.config(text="Model reset - Ready to train")
        self.plot_data()
        self.clear_training_history()
        self.update_stats()
    
    def plot_training_history(self):
        if not self.perceptron or not self.perceptron.training_history['epochs']:
            return
        
        self.history_fig.clear()
        
        # Create subplots
        ax1 = self.history_fig.add_subplot(211)
        ax2 = self.history_fig.add_subplot(212)
        
        epochs = self.perceptron.training_history['epochs']
        errors = self.perceptron.training_history['errors']
        accuracy = self.perceptron.training_history['accuracy']
        
        # Plot errors
        ax1.plot(epochs, errors, 'r-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Error')
        ax1.set_title('Training Error')
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, accuracy, 'b-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        
        self.history_fig.tight_layout()
        self.history_canvas.draw()
    
    def clear_training_history(self):
        self.history_fig.clear()
        self.history_canvas.draw()
    
    def update_stats(self):
        self.stats_text.delete(1.0, tk.END)
        
        X, y = self.get_combined_data()
        if X is None:
            self.stats_text.insert(tk.END, "No data available\n")
            return
        
        stats = f"Dataset Statistics:\n"
        stats += f"Total samples: {len(X)}\n"
        stats += f"Features: {X.shape[1]}\n"
        stats += f"Class 0 samples: {np.sum(y == 0)}\n"
        stats += f"Class 1 samples: {np.sum(y == 1)}\n"
        stats += f"Feature 1 range: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]\n"
        stats += f"Feature 2 range: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]\n\n"
        
        if self.perceptron and self.perceptron.is_trained:
            predictions = self.perceptron.predict(X)
            accuracy = np.mean(predictions == y)
            
            stats += f"Model Statistics:\n"
            stats += f"Final Accuracy: {accuracy:.3f}\n"
            stats += f"Weights: [{self.perceptron.weights[0]:.3f}, {self.perceptron.weights[1]:.3f}]\n"
            stats += f"Bias: {self.perceptron.bias:.3f}\n"
            stats += f"Learning Rate: {self.perceptron.learning_rate}\n"
            stats += f"Training Epochs: {len(self.perceptron.training_history['epochs'])}\n"
            
            if self.perceptron.training_history['epochs']:
                final_error = self.perceptron.training_history['errors'][-1]
                stats += f"Final Training Error: {final_error}\n"
        
        self.stats_text.insert(tk.END, stats)
    
    def export_model(self):
        if not self.perceptron or not self.perceptron.is_trained:
            messagebox.showerror("Error", "No trained model to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".npz",
            filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                np.savez(file_path, 
                        weights=self.perceptron.weights,
                        bias=self.perceptron.bias,
                        learning_rate=self.perceptron.learning_rate)
                messagebox.showinfo("Success", "Model exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export model: {str(e)}")
    
    def export_data(self):
        X, y = self.get_combined_data()
        if X is None:
            messagebox.showerror("Error", "No data to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
                df['target'] = y
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Data exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def export_plot(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", "Plot exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plot: {str(e)}")

# Configure grid weights for responsive design
def configure_grid_weights(root):
    root.grid_rowconfigure(0, weight=3)
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=2)
