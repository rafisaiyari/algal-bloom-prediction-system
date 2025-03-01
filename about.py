import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

class aboutPage(tk.Frame):
    def __init__(self, parent, bg="#F1F1F1"):
        super().__init__(parent, bg=bg)
        self.parent = parent
        self.propagate(False)
        self.create_widgets()
    
    def create_widgets(self):

        aboutlb = tk.Label(self, text="ABOUT US: THE DEVELOPERS", justify="left", anchor="w", font=("Arial", 25, "bold"))
        aboutlb.grid(row=0, column=0, columnspan=4, padx=20, pady=20, sticky="ew")
        self.columnconfigure(0,weight=1) 
        
        self.canvas = tk.Canvas(self, bg="#F1F1F1")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.rowconfigure(0,weight=1)
        self.canvas.columnconfigure(0,weight=1)
        
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.grid(row=0,column=1, sticky="nsew")
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        def on_mousewheel(event):
            self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

        self.canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        self.content = tk.Frame(width=500 , bg="#F1F1F1")
        self.content.grid(row=1, column=0, padx=10, pady=30, sticky="nsew")
        self.content.rowconfigure(0,weight=1)
        self.content.columnconfigure(0,weight=1)
        self.content.rowconfigure(1,weight=1)
        self.content.columnconfigure(1,weight=1)
        self.content.rowconfigure(2,weight=1)
        self.content.columnconfigure(2,weight=1)
        self.content.columnconfigure(3,weight=1) 
        
        self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        
        def make_circle_image(image_path, size=(100, 100)):
            img = Image.open(image_path).resize(size, Image.LANCZOS)  # Resize image
            mask = Image.new("L", size, 0)  # Create a blank mask
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, size[0], size[1]), fill=255)  # Draw a white circle
            img.putalpha(mask)  # Apply mask
            return ImageTk.PhotoImage(img)

        self.P1 = make_circle_image("DevPics/Benj.jpg", size=(150,150))
        self.P2 = make_circle_image("DevPics/Matt.jpg", size=(150,150))
        self.P3 = make_circle_image("DevPics/Rafi.jpg", size=(150,150))
        self.P4 = make_circle_image("DevPics/Beau.jpg", size=(150,150))

        self.DP1 = tk.Label(self.content, image=self.P1, anchor="center")
        self.DP1.grid(row=0, column=0, padx=30, pady=50, sticky="ew")
        self.N1 = tk.Label(self.content, text="Franz Benjamin Africano", bg="#F1F1F1", anchor="center", font=("Arial", 11, "bold"))
        self.N1.grid(row=1, column=0, pady=10, sticky="ew")

        self.DP2 = tk.Label(self.content, image=self.P2, anchor="center")
        self.DP2.grid(row=0, column=1, padx=30, pady=50, sticky="ew")
        self.N2 = tk.Label(self.content, text="Matt Terrence Rias", bg="#F1F1F1", anchor="center", font=("Arial", 11, "bold"))
        self.N2.grid(row=1, column=1, pady=10, sticky="ew")

        self.DP3 = tk.Label(self.content, image=self.P3, anchor="center")
        self.DP3.grid(row=0, column=2, padx=30, pady=50, sticky="ew")
        self.N3 = tk.Label(self.content, text="Rafi Saiyari", bg="#F1F1F1", anchor="center", font=("Arial", 11, "bold"))
        self.N3.grid(row=1, column=2, pady=10, sticky="ew")

        self.DP4 = tk.Label(self.content, image=self.P4, anchor="center")
        self.DP4.grid(row=0, column=3, padx=30, pady=50, sticky="ew")
        self.N4 = tk.Label(self.content, text="Beau Lawyjet Sison", bg="#F1F1F1", anchor="center", font=("Arial", 11, "bold"))
        self.N4.grid(row=1, column=3, pady=10, sticky="ew")

        # Project Information
        self.project_info = (
            """We are Terra! Our team is a dedicated group of four third-year Computer Science students driven by a shared passion for technology and environmental sustainability. Our team—Franz Benjamin Africano, Matt Terrence Rias, Mohammad Rafi Saiyari, and Beau Lawyjet Sison—aims to leverage our technical expertise to develop innovative solutions for real-world environmental challenges.

With the rapid degradation of aquatic ecosystems due to pollution and climate change, we recognize the urgent need for advanced monitoring and predictive systems. Our study, BloomSentry: An Algal Bloom Monitoring and Prediction System, seeks to enhance the Laguna Lake Development Authority’s (LLDA) efforts in water quality management by implementing machine learning-driven analytics. By utilizing Long Short-Term Memory (LSTM) models, we aim to provide accurate, data-driven predictions of algal bloom occurrences, allowing for proactive decision-making and environmental conservation.

Through this research, Team Terra aspires to bridge the gap between computer science and environmental sustainability, contributing to a smarter, data-informed approach to preserving Laguna Lake’s ecological balance."""
        )

        self.project_label = tk.Label(self.content, text=self.project_info, wraplength=1000, justify="center", bg="#F1F1F1", font=("Arial", 12))
        self.project_label.grid(row=2, column=0, columnspan=4, pady=50,padx=20, sticky="ew")

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")


