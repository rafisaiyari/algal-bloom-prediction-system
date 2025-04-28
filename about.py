import customtkinter as ctk
from PIL import Image, ImageDraw


class AboutPage(ctk.CTkFrame):
    def __init__(self, parent, bg_color=None):
        super().__init__(parent, fg_color=bg_color or "transparent")
        self.parent = parent
        self.propagate(False)
        self.create_widgets()

    def create_widgets(self):
        aboutlb = ctk.CTkLabel(self, text="ABOUT US: THE DEVELOPERS", justify="left", anchor="w",
                               font=("Arial", 25, "bold"))
        aboutlb.grid(row=0, column=0, columnspan=4, padx=20, pady=20, sticky="ew")
        self.columnconfigure(0, weight=1)

        # CustomTkinter's scrollable frame
        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=30)
        self.rowconfigure(1, weight=1)

        # Create content inside the scrollable frame
        self.content = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.content.grid(row=0, column=0, padx=10, pady=30, sticky="nsew")
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=1)
        self.content.columnconfigure(1, weight=1)
        self.content.rowconfigure(2, weight=1)
        self.content.columnconfigure(2, weight=1)
        self.content.columnconfigure(3, weight=1)

        def make_circle_image(image_path, size=(100, 100)):
            """Create a circular image using PIL and CTkImage"""
            # Open and resize the image
            img = Image.open(image_path).resize(size, Image.LANCZOS)

            # Create a circular mask
            mask = Image.new("L", size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, size[0], size[1]), fill=255)

            # Apply the mask to create a circular image
            img_rgba = img.convert("RGBA")
            circle_img = Image.new("RGBA", size, (0, 0, 0, 0))
            circle_img.paste(img_rgba, (0, 0), mask)

            # Return as CTkImage
            return ctk.CTkImage(light_image=circle_img, dark_image=circle_img, size=size)

        # Create circular profile images using CTkImage
        self.P1 = make_circle_image("DevPics/Benj.jpg", size=(150, 150))
        self.P2 = make_circle_image("DevPics/Matt.jpg", size=(150, 150))
        self.P3 = make_circle_image("DevPics/Rafi.jpg", size=(150, 150))
        self.P4 = make_circle_image("DevPics/Beau.jpg", size=(150, 150))

        # Create profile displays with CTkLabels
        self.DP1 = ctk.CTkLabel(self.content, image=self.P1, text="")
        self.DP1.grid(row=0, column=0, padx=30, pady=50, sticky="ew")
        self.N1 = ctk.CTkLabel(self.content, text="Franz Benjamin Africano", anchor="center",
                               font=("Arial", 11, "bold"))
        self.N1.grid(row=1, column=0, pady=10, sticky="ew")

        self.DP2 = ctk.CTkLabel(self.content, image=self.P2, text="")
        self.DP2.grid(row=0, column=1, padx=30, pady=50, sticky="ew")
        self.N2 = ctk.CTkLabel(self.content, text="Matt Terrence Rias", anchor="center",
                               font=("Arial", 11, "bold"))
        self.N2.grid(row=1, column=1, pady=10, sticky="ew")

        self.DP3 = ctk.CTkLabel(self.content, image=self.P3, text="")
        self.DP3.grid(row=0, column=2, padx=30, pady=50, sticky="ew")
        self.N3 = ctk.CTkLabel(self.content, text="Rafi Saiyari", anchor="center", font=("Arial", 11, "bold"))
        self.N3.grid(row=1, column=2, pady=10, sticky="ew")

        self.DP4 = ctk.CTkLabel(self.content, image=self.P4, text="")
        self.DP4.grid(row=0, column=3, padx=30, pady=50, sticky="ew")
        self.N4 = ctk.CTkLabel(self.content, text="Beau Lawyjet Sison", anchor="center",
                               font=("Arial", 11, "bold"))
        self.N4.grid(row=1, column=3, pady=10, sticky="ew")

        # Project Information
        self.project_info = (
            """We are Terra! Our team is a dedicated group of four third-year Computer Science students driven by a shared passion for technology and environmental sustainability. Our team—Franz Benjamin Africano, Matt Terrence Rias, Mohammad Rafi Saiyari, and Beau Lawyjet Sison—aims to leverage our technical expertise to develop innovative solutions for real-world environmental challenges.

With the rapid degradation of aquatic ecosystems due to pollution and climate change, we recognize the urgent need for advanced monitoring and predictive systems. Our study, BloomSentry: An Algal Bloom Monitoring and Prediction System, seeks to enhance the Laguna Lake Development Authority's (LLDA) efforts in water quality management by implementing machine learning-driven analytics. By utilizing Long Short-Term Memory (LSTM) models, we aim to provide accurate, data-driven predictions of algal bloom occurrences, allowing for proactive decision-making and environmental conservation.

Through this research, Team Terra aspires to bridge the gap between computer science and environmental sustainability, contributing to a smarter, data-informed approach to preserving Laguna Lake's ecological balance."""
        )

        self.project_label = ctk.CTkLabel(self.content, text=self.project_info, wraplength=1000, justify="center",
                                          font=("Arial", 12))
        self.project_label.grid(row=2, column=0, columnspan=4, pady=50, padx=20, sticky="ew")

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")