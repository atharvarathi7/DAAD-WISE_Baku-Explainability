import tkinter as tk
from tkinter import filedialog, messagebox
import pickle
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from io import BytesIO
import torch
#from generic_attn_map import generic_attn_map

class PickleViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pickle File Viewer")
        self.data = []
        self.frames = []  # List to store frames
        self.proprio = [] # List to store proprio features

        # Open in maximized mode on Ubuntu
        self.root.wm_attributes('-zoomed', True)
        # self.root.grid_rowconfigure(0, weight=1)
        # self.root.grid_coloumnconfigure(0, weight=1)
        # Main frame to hold all widgets
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Entry for pickle file path
        self.file_path_entry = tk.Entry(self.main_frame, width=50)
        self.file_path_entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.file_path_entry.insert(0, 'Enter pickle file path or use "Browse"')

        # Button to browse for file
        self.browse_button = tk.Button(self.main_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=1, padx=10, pady=10)

        # Button to load and display file contents
        self.load_button = tk.Button(self.main_frame, text="Load Pickle File", command=self.load_pickle_file)
        self.load_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Container for dictionary frames
        self.frame_container = tk.Frame(self.main_frame)
        self.frame_container.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Make the frame container expand with window resize
        self.main_frame.grid_rowconfigure(2, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Slider for navigating through data
        self.slider = tk.Scale(self.main_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_display)
        self.slider.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)

    def load_pickle_file(self):
        file_path = self.file_path_entry.get()
        if not file_path:
            messagebox.showerror("Error", "Please enter a file path or browse for a file.")
            return
        
        try:
            with open(file_path, 'rb') as file:
                self.data = pickle.load(file)
            
            # Check if the data is a list of dictionaries
            if not (isinstance(self.data, list) and all(isinstance(item, dict) for item in self.data)):
                raise ValueError("Pickle file does not contain a list of dictionaries.")
            
            # Clear the previous frames
            for frame in self.frames:
                frame.destroy()
            self.frames = []

            for item in self.data:
                for key, val in item.items():
                        if key == 'Observation':
                            value = val['proprioceptive']
                            self.proprio.append(value)

            # Create a frame for each dictionary
            for item in self.data:
                frame = tk.Frame(self.frame_container)
                frame.pack(fill=tk.BOTH, expand=True)
                
                # Display image, heatmap, and a third image in the same row
                if "Agent View" in item and "attn_maps" in item and "Observation" in item:

                    image_label = self.display_image(frame, item["Agent View"])
                    image_label.grid(row=0, column=0, padx=20, pady=10)
                    
                    #print(item["attn_maps"].requires_grad)
                    heatmap_label = self.display_heatmap(frame, item["attn_maps"])
                    heatmap_label.grid(row=0, column=1, padx=10, pady=10)
                    
                    plot_label = self.display_plot(frame, self.proprio)
                    plot_label.grid(row=0, column=2, padx=10, pady=10)
                    
                    # Labels for image, heatmap, and plot
                    image_text_label = tk.Label(frame, text="Agent View")
                    image_text_label.grid(row=1, column=0)

                    heatmap_text_label = tk.Label(frame, text="Attention Heatmap")
                    heatmap_text_label.grid(row=1, column=1)

                    plot_text_label = tk.Label(frame, text="Generated Plot")
                    plot_text_label.grid(row=1, column=2)

                    # Text widget to display dictionary content
                    text_widget = tk.Text(frame, width=80, height=10)
                    text_widget.grid(row=2, column=0, columnspan=3, padx=10, pady=10)
                    for key, val in item.items():
                        if key == 'Observation':
                            value = val['proprioceptive']
                            text_widget.insert(tk.END, f"Proprioceptive Features: {value}\n")
                
                self.frames.append(frame)

            # Set slider parameters
            self.slider.config(to=len(self.frames) - 1)

            # Display the first frame
            self.update_display(0)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load pickle file: {e}")

    def display_plot(self, parent, data):

        data = np.array(data)    
        fig, ax = plt.subplots()
        for i in range(data.shape[1]):
            ax.plot(data[:, i], label=f'Vector {i+1}')
            
        #ax.legend()
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Proprioceptive Values')
        ax.set_title('Proprioceptive Values Plot')

        #canvas = FigureCanvasTkAgg(fig, master=parent)
        fig.canvas.draw()
        #canvas.get_tk_widget().pack(fill=tk.BOTH, expand = True)

        # Convert plot to an image in memory
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        # Create an image from the buffer
        plot = Image.open(buf)
        #resized_plot = plot.resize((256, 256), Image.LANCZOS)

        photo = ImageTk.PhotoImage(plot)
        buf.close()

        # Create a label to display the heatmap
        plot_label = tk.Label(parent, image=photo)
        plot_label.image = photo  # Keep a reference to avoid garbage collection
        
        return plot_label

    
    def display_image(self, parent, pixel_data):
        # Assuming pixel_data is a flat list of pixel values for a 256x256 image
        pixel_data_np = np.array(pixel_data, dtype=np.uint8).reshape((256, 256, 3))
        image = Image.fromarray(pixel_data_np)

        # Convert image to PhotoImage
        photo = ImageTk.PhotoImage(image)
        
        # Create a label to display the image
        image_label = tk.Label(parent, image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
        
        return image_label

    def display_heatmap(self, parent, attention_data):
        #print(attention_data)
        #attention_data = attention_data[0]
        R = torch.eye(5, 5).cuda()
        R = R.unsqueeze(0).expand(1, 5, 5)
        #print(R)
        for i, blk in enumerate(attention_data):
            #print(i,blk)
            cam = blk.detach()
            
            #print(cam)
            cam = cam.clamp(min=0).mean(dim=1)
            #print(cam)
            R = R + torch.bmm(cam, R)
            #print(R)
        attention_relevance = R[0]
        #attention_data = attention_data.clamp(min=0).mean(dim=1)
        #print(R)

        attention_relevance = attention_relevance.cpu().numpy()
        attention_relevance = attention_relevance / attention_relevance.max()
        #print(attention_relevance)
        #attention_data = generic_attn_map(action, attention_data)
        # Generate heatmap from 5x5 self-attention data
        fig, ax = plt.subplots()
        cax = ax.matshow(attention_relevance, cmap='viridis')
        fig.colorbar(cax)

        # Resize the figure before saving to reduce image size
        fig.tight_layout(pad=1.0)
        fig.canvas.draw()

        # Convert plot to an image in memory
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        # Create an image from the buffer
        heatmap_image = Image.open(buf)
        #resized_heatmap_image = heatmap_image.resize((256, 256), Image.LANCZOS)

        photo = ImageTk.PhotoImage(heatmap_image)
        buf.close()

        # Create a label to display the heatmap
        heatmap_label = tk.Label(parent, image=photo)
        heatmap_label.image = photo  # Keep a reference to avoid garbage collection
        
        return heatmap_label

    def update_display(self, value):
        index = int(value)
        if 0 <= index < len(self.frames):
            for frame in self.frames:
                frame.pack_forget()  # Hide all frames
            self.frames[index].pack(fill=tk.X, expand=True)  # Show the current frame
        else:
            messagebox.showerror("Error", "Index out of range.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PickleViewerApp(root)
    root.mainloop()
