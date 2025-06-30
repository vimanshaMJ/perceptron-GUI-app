import tkinter as tk
from tkinter import ttk
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.gui.main_window import PerceptronGUI, configure_grid_weights

def main():
    # Create main window
    root = tk.Tk()
    
    # Configure grid weights for responsive design
    configure_grid_weights(root)
    
    # Set window icon (if available)
    try:
        root.iconbitmap('assets/icon.ico')
    except:
        pass  # Icon file not found, continue without it
    
    # Create and run application
    app = PerceptronGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
