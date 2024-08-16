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
    def __init__(self, frame, img_array=None, plot_func=None, data=None, video_path=None):
        self.frame = frame
        self.img_array = img_array
        self.plot_func = plot_func
        self.video_path = video_path
        self.canvas = tk.Canvas(frame, highlightthickness=0)
        self.canvas.grid(sticky='nsew')
        self.canvas.bind('<Configure>', self.resize_content)
        self.photo = None
        self.video_cap = None
        self.data = data

        if self.video_path:
            self.init_video()

    def resize_content(self, event):
        width, height = event.width, event.height
        if self.img_array is not None:
            self.resize_image(width, height)
        elif self.plot_func is not None:
            self.resize_plot(width, height)

    def resize_image(self, width, height):
        resized_image = Image.fromarray(self.img_array).resize((width, height), Image.LANCZOS)
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

    def init_video(self):
        self.video_cap = cv2.VideoCapture(self.video_path)
        self.update_video_frame()

    def update_video_frame(self):
        if self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                self.photo = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
                self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
                # Schedule the next frame update
                self.frame.after(30, self.update_video_frame)
            else:
                self.video_cap.release()

def create_resizable_content(frame, img_array, plot_func, data, video_path, row, col):
    resizable_content = ResizableImage(frame, img_array, plot_func, data, video_path)
    resizable_content.canvas.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
    frame.grid_rowconfigure(row, weight=1)
    frame.grid_columnconfigure(col, weight=1)

def plot_func1(ax, data):
    data = np.array(data)    
    for i in range(data.shape[1]):
        ax.plot(data[:, i], label=f'Vector {i+1}')
    ax.legend()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Proprioceptive Values')
    ax.set_title('Proprioceptive Values Plot')

def plot_func2(ax, data):
    data = np.array(data)    
    for i in range(data.shape[1]):
        ax.plot(data[:, i], label=f'Vector {i+1}')
    ax.legend()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Action Values')
    ax.set_title('Action Values Plot')

def plot_func3(ax, attention_data):
    R = torch.eye(5, 5).cuda()
    R = R.unsqueeze(0).expand(1, 5, 5)
    for i, blk in enumerate(attention_data):
        cam = blk.detach()
         # equation X 
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    attention_relevance = R[0]
    attention_relevance = attention_relevance.cpu().numpy()
    attention_relevance = attention_relevance / attention_relevance.max()
    #print(attention_relevance)
    # Generate heatmap from 5x5 self-attention data
    ax.imshow(attention_relevance, cmap='viridis')

def plot_func4(ax, attention_data):
    attention_data = attention_data.cpu().numpy()
    print(attention_data)
    ax.imshow(attention_data, cmap='viridis')

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
    if file_path:
        file_path_entry.delete(0, tk.END)
        file_path_entry.insert(0, file_path)

def load_pickle_file():
    global frames2
    file_path = file_path_entry.get()
    video_path = "/home/atharva/BAKU/baku/exp_local/eval/2024.07.31_eval/deterministic/173915_hidden_dim_256/eval_video/0_env0.mp4"
    
    
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
        
        # Check if the data is a list of dictionaries
        if not (isinstance(data, list) and all(isinstance(item, dict) for item in data)):
            raise ValueError("Pickle file does not contain a list of dictionaries.")
        
        # Clear the previous frames
        frames2 = []
        proprio = []
        action = []
        for item in data:
            for key, val in item.items():
                if key == 'Observation':
                    pro = val['proprioceptive']
                    proprio.append(pro)
                        
                elif key == 'Action':
                    action.append(val)

        # Create a frame for each dictionary
        for item in data:
            frame2 = ttk.Frame(frame_container)
            frame1 = ttk.Frame(frame_container)

            # Display two plots in frame1 (first row) and initialize video in frame1 (second row)
            create_resizable_content(frame1, None, plot_func1, proprio, None, 0, 0)
            create_resizable_content(frame1, None, plot_func2, action, None, 0, 1)

            # Initialize video player to occupy the second row of frame1
            create_resizable_content(frame1, None, None, None, video_path, 1, 0)

            # Display four plots in frame2 in a 2x2 grid
            #print(item["attn_maps"][-1][-1][-1])
            create_resizable_content(frame2, None, plot_func3, item["attn_maps"], None, 0, 0)
            create_resizable_content(frame2, None, plot_func4, item["attn_maps"][-1][-1][-1], None, 0, 1)
            create_resizable_content(frame2, item['Agent View'], None, None, None, 1, 0)
            create_resizable_content(frame2, item['Agent View'], None, None, None, 1, 1)

            print(item['Agent View'])
            import pdb; pdb.set_trace()
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
data =[]

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