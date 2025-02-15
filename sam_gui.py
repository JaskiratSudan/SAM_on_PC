import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import threading
from segment_anything import sam_model_registry, SamPredictor

# Load SAM Model
model_type = "vit_h"
checkpoint_path = "SAM Vit Model.pth"  # Update path if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
predictor = SamPredictor(sam)


class SAM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM Segmentation GUI")
        self.root.geometry("800x600")

        self.image_path = None
        self.original_image = None
        self.points = []
        self.bboxes = []

        # UI Components
        self.btn_load = tk.Button(root, text="Load Image", command=self.load_image)
        self.btn_load.pack()

        self.canvas = tk.Canvas(root, width=600, height=400, bg="gray")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.add_point)

        self.lbl_info = tk.Label(root, text="Click to add points or enter bounding box below:")
        self.lbl_info.pack()

        # Input Fields for Bounding Box
        self.entry_bbox = tk.Entry(root, width=40)
        self.entry_bbox.pack()
        self.entry_bbox.insert(0, "x1,y1,x2,y2")  # Placeholder text

        self.entry_text_prompt = tk.Entry(root, width=40)
        self.entry_text_prompt.pack()
        self.entry_text_prompt.insert(0, "Enter Text Prompt (optional)")

        # Run SAM Button
        self.btn_run = tk.Button(root, text="Run SAM", command=self.run_sam_thread)
        self.btn_run.pack()

        # Progress Bar
        self.progress = ttk.Progressbar(root, orient="horizontal", mode="indeterminate", length=300)
        self.progress.pack(pady=10)

        self.btn_clear = tk.Button(root, text="Clear Points", command=self.clear_points)
        self.btn_clear.pack()

    def load_image(self):
        """Loads an image from file and displays it on the canvas."""
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not self.image_path:
            return

        self.original_image = cv2.imread(self.image_path)
        self.display_image(self.original_image)

    def display_image(self, image):
        """Displays an image on the Tkinter canvas."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 400), Image.LANCZOS)  # Fixed .ANTIALIAS issue
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def add_point(self, event):
        """Captures points clicked on the canvas."""
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red")
        print(f"Point added: {x}, {y}")

    def run_sam_thread(self):
        """Starts a new thread to run SAM without freezing the UI."""
        self.progress.start()
        threading.Thread(target=self.run_sam, daemon=True).start()

    def run_sam(self):
        """Runs SAM on the loaded image using user-defined points, bounding boxes, and text prompts."""
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded.")
            self.progress.stop()
            return

        # Convert user-defined points to NumPy format
        input_points = np.array(self.points) if self.points else None
        input_labels = np.ones(len(self.points)) if self.points else None  # Assume foreground labels

        # Parse bounding box input
        bbox_text = self.entry_bbox.get()
        if bbox_text != "x1,y1,x2,y2":
            try:
                x1, y1, x2, y2 = map(int, bbox_text.split(","))
                self.bboxes = np.array([[x1, y1, x2, y2]])
            except ValueError:
                messagebox.showerror("Error", "Invalid bounding box format. Use: x1,y1,x2,y2")
                self.progress.stop()
                return

        # Prepare image for SAM
        predictor.set_image(self.original_image)

        # Run SAM Prediction
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=self.bboxes if self.bboxes else None,
            multimask_output=False
        )

        # Overlay mask on the image
        mask = masks[0]
        overlay = self.original_image.copy()
        overlay[mask] = [0, 255, 0]  # Green overlay for mask
        self.display_image(overlay)

        self.progress.stop()  # Stop progress bar after inference

    def clear_points(self):
        """Clears user-selected points and bounding boxes."""
        self.points = []
        self.bboxes = []
        self.canvas.delete("all")
        if self.original_image is not None:
            self.display_image(self.original_image)


# Run GUI
root = tk.Tk()
app = SAM_GUI(root)
root.mainloop()