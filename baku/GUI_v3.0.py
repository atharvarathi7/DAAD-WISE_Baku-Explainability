import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2
import pickle
import torch

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
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def resize_plot(self, width, height):
        fig = Figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111)
        self.plot_func(ax, self.data)

        # Clear the previous plot
        self.canvas.delete("all")

        canvas_agg = FigureCanvasTkAgg(fig, self.canvas)
        canvas_agg.draw()

        # Convert canvas to an image
        self.photo = ImageTk.PhotoImage(master=self.canvas, image=Image.fromarray(np.array(canvas_agg.get_renderer()._renderer)))
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

def create_resizable_content(frame, img_array, plot_func, data, row, col):
    resizable_content = ResizableImage(frame, img_array, plot_func, data)
    resizable_content.canvas.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
    frame.grid_rowconfigure(row, weight=1)
    frame.grid_columnconfigure(col, weight=1)

def plot_func1(ax, data):
    data = np.array(data)   
    for i in range(7):  
        ax.plot(data[:, i], label=f'Vector {i+1}')
    ax.legend()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Proprioceptive Values')
    ax.set_title('Proprioceptive Values Plot')

def plot_func2(ax, data):
    data = np.array(data)    
    for i in range(7):
        ax.plot(data[:, i], label=f'Vector {i+1}')
    ax.legend()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Action Values')
    ax.set_title('Action Values Plot')


def plot_func4(ax, attention_data):
    attention_data = np.array(attention_data)
    #print(attention_data)
    ax.imshow(attention_data, cmap='viridis')

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
    if file_path:
        file_path_entry.delete(0, tk.END)
        file_path_entry.insert(0, file_path)

def load_pickle_file():
    global frames2
    file_path = file_path_entry.get()

    # Store data in Backend 

    # load first timestep data and plot!
    # make canvas
    # --> update canvases indivudually 
    
    
    if not file_path:
        messagebox.showerror("Error", "Please enter a file path or browse for a file.")
        return
    
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # Clear the previous frames
        frames2 = []
        frame1 = ttk.Frame(frame_container)
        create_resizable_content(frame1, None, plot_func1, data["proprioceptive"], 0, 0)
        create_resizable_content(frame1, None, plot_func2, data["action"], 0, 1)
        #print(data)
        #import pdb; pdb.set_trace()
        # Create a frame for each dictionary
        for i in range(340):
            frame2 = ttk.Frame(frame_container)
            # Display two plots in frame1 (first row) and initialize video in frame1 (second row)
            # Display four plots in frame2 in a 2x2 grid
            create_resizable_content(frame2, None, plot_func4, data["generic_atten"][i], 0, 0)
            create_resizable_content(frame2, None, plot_func4, data["atten_maps"][i][-1][-1][-1], 0, 1)
            create_resizable_content(frame2, data["pixels"][i], None, None, 1, 0)
            create_resizable_content(frame2, data["pixels_egocentric"][i], None, None, 1, 1)
            frames2.append(frame2)

        # Set slider parameters
        slider.config(to=len(frames2) - 1)

        # Display the first frame
        frame1.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        update_display(0)
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load pickle file: {e}")

def update_display(value):
    global frames2
    index = int(value)
    if 0 <= index < len(frames2):
        for frame2 in frames2:
            frame2.grid_forget()  # Hide all frames
        frames2[index].grid(row=0, column=1, padx=10, pady=10, sticky='nsew')  # Show the current frame
    else:
        messagebox.showerror("Error", "Index out of range.")

# Create the main window
root = tk.Tk()
root.title("Generic Explainability GUI")
root.wm_attributes('-zoomed', True)

# Create a main container frame to hold all widgets
main_frame = ttk.Frame(root)
main_frame.pack(padx=10, pady=10, fill='both', expand=True)

# Container for dictionary frames
frame_container = ttk.Frame(main_frame)
frame_container.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Create three frames within the main container frame
# frame1 = ttk.Frame(frame_container)
# frame1.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

frame_container.columnconfigure(0, weight=1)
frame_container.columnconfigure(1, weight=1)
frame_container.rowconfigure(0, weight=1)

# Allow frame1 and frame2 to expand to fill the available space
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(2, weight=1)

# Entry for file path
file_path_entry = ttk.Entry(main_frame, width=50)
file_path_entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
file_path_entry.insert(0, 'Enter pickle file path or use "Browse"')

# Button to browse for file
browse_button = ttk.Button(main_frame, text="Browse", command= browse_file)
browse_button.grid(row=0, column=1, padx=10, pady=10)

# Button to load and display file contents
load_button = ttk.Button(main_frame, text="Load Pickle File", command=load_pickle_file)
load_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Slider for navigating across timesteps
slider = tk.Scale(main_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=update_display)
slider.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

# Start the Tkinter main loop
root.mainloop()