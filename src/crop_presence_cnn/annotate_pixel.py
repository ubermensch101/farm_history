import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ColorMaskApp:
    def __init__(self, root, image_paths):
        self.root = root
        self.root.title("Color Masking")
        self.image_paths = image_paths
        self.current_image_index = 0

        # Dictionary containing default color threshold values
        self.color_thresh = {
            "low_red": np.array([5, 100, 60]),
            "upper_red": np.array([20, 180, 200]),
            "low_green": np.array([15, 80, 40]),
            "upper_green": np.array([50, 180, 100]),
        }

        # Initialize variables for color thresholds with default values
        self.hue_low_color1 = tk.StringVar(value=str(self.color_thresh["low_red"][0]))
        self.hue_high_color1 = tk.StringVar(value=str(self.color_thresh["upper_red"][0]))
        self.saturation_low_color1 = tk.StringVar(value=str(self.color_thresh["low_red"][1]))
        self.saturation_high_color1 = tk.StringVar(value=str(self.color_thresh["upper_red"][1]))
        self.value_low_color1 = tk.StringVar(value=str(self.color_thresh["low_red"][2]))
        self.value_high_color1 = tk.StringVar(value=str(self.color_thresh["upper_red"][2]))

        self.hue_low_color2 = tk.StringVar(value=str(self.color_thresh["low_green"][0]))
        self.hue_high_color2 = tk.StringVar(value=str(self.color_thresh["upper_green"][0]))
        self.saturation_low_color2 = tk.StringVar(value=str(self.color_thresh["low_green"][1]))
        self.saturation_high_color2 = tk.StringVar(value=str(self.color_thresh["upper_green"][1]))
        self.value_low_color2 = tk.StringVar(value=str(self.color_thresh["low_green"][2]))
        self.value_high_color2 = tk.StringVar(value=str(self.color_thresh["upper_green"][2]))

        self.init_gui()
        self.load_image()

    def init_gui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0)

        # Color 1
        ttk.Label(self.main_frame, text="Not Crop").grid(row=0, column=0, columnspan=3)
        self.create_entry("Hue Low", self.hue_low_color1, row=1, column=0)
        self.create_entry("Hue High", self.hue_high_color1, row=2, column=0)
        self.create_entry("Saturation Low", self.saturation_low_color1, row=3, column=0)
        self.create_entry("Saturation High", self.saturation_high_color1, row=4, column=0)
        self.create_entry("Value Low", self.value_low_color1, row=5, column=0)
        self.create_entry("Value High", self.value_high_color1, row=6, column=0)

        # Empty columns between Color 1 and Color 2
        ttk.Label(self.main_frame, text="").grid(row=0, column=3)
        ttk.Label(self.main_frame, text="").grid(row=1, column=3)
        ttk.Label(self.main_frame, text="").grid(row=2, column=3)
        ttk.Label(self.main_frame, text="").grid(row=3, column=3)
        ttk.Label(self.main_frame, text="").grid(row=4, column=3)
        ttk.Label(self.main_frame, text="").grid(row=5, column=3)
        ttk.Label(self.main_frame, text="").grid(row=6, column=3)

        # Color 2
        ttk.Label(self.main_frame, text="Crop (Green)").grid(row=0, column=4, columnspan=3)
        self.create_entry("Hue Low", self.hue_low_color2, row=1, column=4)
        self.create_entry("Hue High", self.hue_high_color2, row=2, column=4)
        self.create_entry("Saturation Low", self.saturation_low_color2, row=3, column=4)
        self.create_entry("Saturation High", self.saturation_high_color2, row=4, column=4)
        self.create_entry("Value Low", self.value_low_color2, row=5, column=4)
        self.create_entry("Value High", self.value_high_color2, row=6, column=4)

        # Load and Next Image buttons
        ttk.Button(self.main_frame, text="Load Image", command=self.load_image).grid(row=15, column=0, columnspan=2)
        ttk.Button(self.main_frame, text="Next Image", command=self.next_image).grid(row=15, column=2, columnspan=2)

        # Canvas to display images
        self.canvas = tk.Canvas(self.root, width=300, height=300)
        self.canvas.grid(row=1, column=0, columnspan=7)

        # Label to display image index
        self.image_index_label = tk.Label(self.root, text="Image Index: 0")
        self.image_index_label.grid(row=2, column=0, columnspan=2)

    def create_entry(self, text, var, row, column):
        ttk.Label(self.main_frame, text=text).grid(row=row, column=column, sticky=tk.W)
        entry = ttk.Entry(self.main_frame, textvariable=var)
        entry.grid(row=row, column=column+1, sticky=tk.W)

    def load_image(self):
        self.image = cv2.imread(self.image_paths[self.current_image_index])
        self.image = self.resize_image(self.image, (256, 256))  # Resize image to 256x256
        self.display_images()
        self.image_index_label.config(text=f"Image Index: {self.current_image_index}")

    def next_image(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.load_image()

    def display_images(self):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        color1_mask = self.create_mask(hsv_image, 0)
        color2_mask = self.create_mask(hsv_image, 1)

        original_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        original_image = Image.fromarray(original_image)
        hsv_image = Image.fromarray(hsv_image)
        color2_mask = Image.fromarray(color2_mask)
        color2_mask = Image.fromarray((255-color1_mask))
        color1_mask = Image.fromarray(color1_mask)
        original_image = ImageTk.PhotoImage(original_image.resize((300, 300)))
        hsv_image = ImageTk.PhotoImage(hsv_image.resize((300, 300)))
        color1_mask = ImageTk.PhotoImage(color1_mask.resize((300, 300)))
        color2_mask = ImageTk.PhotoImage(color2_mask.resize((300, 300)))

        # Set canvas size
        self.canvas.config(width=720, height=720)

        # Clear previous images on canvas
        self.canvas.delete("all")
        label = "N" if "/n/" in self.image_paths[self.current_image_index] else "Y"
        # Display images on canvas with labels
        self.canvas.create_text(150, 10, anchor=tk.N, text=f"Original Image - {label}")
        self.canvas.create_image(0, 30, anchor=tk.NW, image=original_image)
        
        self.canvas.create_text(450, 10, anchor=tk.N, text="HSV Image")
        self.canvas.create_image(300, 30, anchor=tk.NW, image=hsv_image)
        
        self.canvas.create_text(150, 400, anchor=tk.N, text="Red Mask")
        self.canvas.create_image(0, 420, anchor=tk.NW, image=color1_mask)
        
        self.canvas.create_text(450, 400, anchor=tk.N, text="Green Mask")
        self.canvas.create_image(300, 420, anchor=tk.NW, image=color2_mask)

        self.root.mainloop()

    def create_mask(self, hsv_image, color_index):
        if color_index == 0:
            lower_color = np.array([int(self.hue_low_color1.get()), int(self.saturation_low_color1.get()), int(self.value_low_color1.get())])
            upper_color = np.array([int(self.hue_high_color1.get()), int(self.saturation_high_color1.get()), int(self.value_high_color1.get())])
        else:
            lower_color = np.array([int(self.hue_low_color2.get()), int(self.saturation_low_color2.get()), int(self.value_low_color2.get())])
            upper_color = np.array([int(self.hue_high_color2.get()), int(self.saturation_high_color2.get()), int(self.value_high_color2.get())])

        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        return mask

    def resize_image(self, image, new_size):
        return cv2.resize(image, new_size)

root = tk.Tk()
image_paths = []
image_paths.extend([os.path.join("data","crop_presence", "test", "y", path) for path in os.listdir(os.path.join("data" ,"crop_presence", "test","y"))])
image_paths.extend([os.path.join("data","crop_presence", "test", "n", path) for path in os.listdir(os.path.join("data" ,"crop_presence", "test","n"))])

app = ColorMaskApp(root, image_paths)
