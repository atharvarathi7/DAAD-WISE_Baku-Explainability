import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2
import pickle
import torch
import os

class ResizableImage:
    def __init__(self, frame, img_array=None, plot_func=None, data=None):
        self.frame = frame
        self.img_array = img_array
        self.plot_func = plot_func
        self.canvas = tk.Canvas(frame, highlightthickness=0)
        self.canvas.grid(sticky='nsew')
        self.canvas.bind('<Configure>', self.resize_content)
        self.photo = None
        self.data = data

    def resize_content(self, event):
        width, height = event.width, event.height
        if self.img_array is not None:
            self.resize_image(width, height)
        elif self.plot_func is not None:
            self.resize_plot(width, height)

    def resize_image(self, width, height):
        image_data = np.transpose(self.img_array, (1, 2, 0))
        resized_image = Image.fromarray(image_data).resize((width, height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def resize_plot(self, width, height):
        fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111)
        self.plot_func(fig, ax, self.data)

        # Clear the previous plot
        self.canvas.delete("all")

        canvas_agg = FigureCanvasTkAgg(fig, self.canvas)
        canvas_agg.draw()

        # Convert canvas to an image
        self.photo = ImageTk.PhotoImage(master=self.canvas, image=Image.fromarray(np.array(canvas_agg.get_renderer()._renderer)))
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

def create_resizable_content(frame, img_array, plot_func, data, row, col, text):
    resizable_content = ResizableImage(frame, img_array, plot_func, data)
    resizable_content.canvas.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
    image_label = tk.Label(frame, text= text)
    image_label.grid(row=row+1, column=col)
    frame.grid_rowconfigure(row, weight=1)
    frame.grid_columnconfigure(col, weight=1)

def plot_func1(fig, ax, data):
    data = np.array(data)   
    for i in range(9):  
        ax.plot(data[:, i], label=f'Vector {i+1}')
    ax.legend()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Proprioceptive Values')
    ax.set_title('Proprioceptive Values Plot')

def plot_func2(fig, ax, data):
    data = np.array(data)    
    for i in range(7):
        ax.plot(data[:, i], label=f'Vector {i+1}')
    ax.legend()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Action Values')
    ax.set_title('Action Values Plot')


def plot_func4(fig, ax, attention_data):
    attention_data = np.array(attention_data)
    cax = ax.matshow(attention_data, cmap='viridis')
    fig.colorbar(cax)
    row_labels = ['Text', 'Pix', 'Ego Pix', 'Proprio', 'Action']
    col_labels = ['Text', 'Pix', 'Ego Pix', 'Proprio', 'Action']

    # Set the labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))

    ax.set_xticklabels(col_labels, rotation=90)  # Rotate column labels for better readability
    ax.set_yticklabels(row_labels)
    fig.tight_layout()
    fig.canvas.draw()

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
    if file_path:
        file_path_entry.delete(0, tk.END)
        file_path_entry.insert(0, file_path)
        update_combobox(file_path)

def update_combobox(folder_path):
    try:
        contents = os.listdir(folder_path)
        combobox['values'] = contents
        if contents:
            combobox.set(contents[0])  # Set the first item as the default selection
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read folder contents: {e}")

def load_pickle_file():
    global data
    file_path = file_path_entry.get()

    # Store data in Backend 

    # load first timestep data and plot!    
    if not file_path:
        messagebox.showerror("Error", "Please enter a file path or browse for a file.")
        return
    
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        create_resizable_content(frame3, None, plot_func1, data["proprioceptive"], 0, 0,"Propceptive Features across Timesteps")
        create_resizable_content(frame3, None, plot_func2, data["action"], 0, 1, "Action Features across Timesteps")
        text_widget.insert(tk.END, f"\n Task: {data['task']}\n")
        text_widget.insert(tk.END, f"\n Agent: {data['agent']}\n")
        text_widget.insert(tk.END, f"\n Suite: {data['suite']}\n")
        text_widget.insert(tk.END, f"\n Goal Achieved: {data['goal_achieved']}\n")
        # Set slider parameters
        slider.config(to=len(data['action']) - 1)

        # Display the first frame
        update_display(0)
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load pickle file: {e}")

def update_display(value):
    global data
    i = int(value)
    create_resizable_content(frame1, None, plot_func4, data["generic_atten"][i], 0, 0,"Generic Attention Map across all layers")
    create_resizable_content(frame1, None, plot_func4, data["atten_maps"][i], 0, 1,"Last Layer Attention Map")
    create_resizable_content(frame2, data["pixels"][i], None, None, 1, 0,"Third Person View")
    create_resizable_content(frame2, data["pixels_egocentric"][i], None, None, 1, 1, "Agent View")

    frame1.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
    frame2.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')  
    frame3.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
    text_widget.grid(row=1, column=0, padx=5, pady=5, sticky = 'nsew')

def save_frame_container_as_image():
    # Get the size and position of the frame_container
    x = frame_container.winfo_rootx()
    y = frame_container.winfo_rooty()
    width = frame_container.winfo_width()
    height = frame_container.winfo_height()

    # Capture the image
    image = ImageGrab.grab(bbox=(x, y, x + width, y + height))

    # Ask for save path and save the image
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if save_path:
        image.save(save_path)
        messagebox.showinfo("Save Successful", f"Frame container saved as {save_path}")

# Create the main window
root = tk.Tk()
root.title("Generic Explainability GUI")
root.wm_attributes('-zoomed', True)
# Create a main container frame to hold all widgets
main_frame = ttk.Frame(root)
main_frame.pack(padx=5, pady=5, fill='both', expand=True)

# Container for dictionary frames
frame_container = ttk.Frame(main_frame)
frame_container.grid(row=2, column=0, columnspan=2, pady=5, sticky="nsew")

# Create three frames within the main container frame
frame1 = ttk.Frame(frame_container, relief= tk.GROOVE)
frame2 = ttk.Frame(frame_container, relief= tk.GROOVE)
frame3 = ttk.Frame(frame_container, relief= tk.GROOVE)
text_widget = tk.Text(frame_container, relief= tk.GROOVE)

frame_container.columnconfigure(0, weight=1)
frame_container.columnconfigure(1, weight=1)
frame_container.rowconfigure(0, weight=1)
#frame_container.rowconfigure(1, weight=1)

# Allow frame1 and frame2 to expand to fill the available space
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(2, weight=1)

# Entry for file path
file_path_entry = ttk.Entry(main_frame, width=50)
file_path_entry.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
file_path_entry.insert(0, 'Enter pickle file path or use "Browse"')

# Combobox to display folder contents
combobox = ttk.Combobox(main_frame, width=50)
combobox.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

# Button to browse for file
browse_button = ttk.Button(main_frame, text="Browse", command= browse_file)
browse_button.grid(row=0, column=1, padx=10, pady=5)

# Button to load and display file contents
load_button = ttk.Button(main_frame, text="Load Pickle File", command=load_pickle_file)
load_button.grid(row=1, column=1, padx=5, pady=5)

# Button to save the frame_container as an image
save_button = ttk.Button(main_frame, text="Save as PNG", command=save_frame_container_as_image)
save_button.grid(row=1, column=2, padx=5, pady=5)

# Slider for navigating across timesteps
slider = tk.Scale(main_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=update_display)
slider.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky='ew')

# Start the Tkinter main loop
root.mainloop()