import customtkinter as ctk
from PIL import Image, ImageDraw


class AboutPage(ctk.CTkFrame):
    def __init__(self, parent, bg_color=None):
        super().__init__(parent, fg_color=bg_color or "transparent")
        self.parent = parent
        self.propagate(False)
        self.create_widgets()
        self.grid_columnconfigure(0, weight=1)

    def create_widgets(self):
        aboutlb = ctk.CTkLabel(
            self, text="ABOUT US: THE DEVELOPERS", 
            justify="left", anchor="w",
            font=("Roboto", 25, "bold")
        )
        aboutlb.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")

        # Scrollable frame for content
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 20))
        self.content_frame.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        def make_circle_image(image_path, size=(80, 80)):
            """Create a circular image using PIL and CTkImage"""
            img = Image.open(image_path).resize(size, Image.LANCZOS)
            mask = Image.new("L", size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, size[0], size[1]), fill=255)

            img_rgba = img.convert("RGBA")
            circle_img = Image.new("RGBA", size, (0, 0, 0, 0))
            circle_img.paste(img_rgba, (0, 0), mask)

            return ctk.CTkImage(light_image=circle_img, dark_image=circle_img, size=size)

        # Developer Images
        self.P1 = make_circle_image("DevPics/Benj.jpg")
        self.P2 = make_circle_image("DevPics/Matt.jpg")
        self.P3 = make_circle_image("DevPics/Rafi.jpg")
        self.P4 = make_circle_image("DevPics/Beau.jpg")

        # Developer Info
        devs = [
            (self.P1, "Franz Benjamin Africano"),
            (self.P2, "Matt Terrence Rias"),
            (self.P3, "Rafi Saiyari"),
            (self.P4, "Beau Lawyjet Sison")
        ]

        for idx , (profile_pic, name) in enumerate(devs):
            card = ctk.CTkFrame(
                self.content_frame, 
                corner_radius=15, 
                fg_color="#ffffff"
            )
            card.grid(
                row=idx+1, column=0, padx=20, pady=10, sticky="nsew"
            )
            card.rowconfigure(idx, weight=1)

            img_label = ctk.CTkLabel(card, image=profile_pic, text="")
            img_label.pack(pady=(20, 10))

            name_label = ctk.CTkLabel(
                card, text=name, font=("Roboto", 13, "bold"), anchor="center"
            )
            name_label.pack(pady=(5, 10))

        # Stretch rows/columns properly
        for i in range(2):
            self.content_frame.rowconfigure(i, weight=1)
            self.content_frame.columnconfigure(i, weight=1)

        # Project Description
        self.project_info = (
            "We are Team Terra, a passionate and dynamic group of four third-year Computer Science students dedicated to "
            "harnessing technology for the betterment of our environment. Our team consists of Franz Benjamin Africano, Matt "
            "Terrence Rias, Mohammad Rafi Saiyari, and Beau Lawyjet Sison. United by a shared vision, we aim to tackle some "
            "of the most pressing environmental challenges of our time using our skills in computer science and data analytics.\n\n"
            
            "Our journey begins with a recognition of the rapid degradation of aquatic ecosystems, especially freshwater lakes, "
            "driven by pollution, overfishing, and the accelerating impacts of climate change. These environmental challenges "
            "threaten not only biodiversity but also the livelihoods of communities that depend on these ecosystems for sustenance "
            "and water supply. In response to this, our team has undertaken a project that combines cutting-edge machine learning "
            "techniques with environmental science to help mitigate the consequences of one such environmental crisis — algal blooms.\n\n"
            
            "Our research project, titled BloomSentry: An Algal Bloom Monitoring and Prediction System, seeks to revolutionize "
            "the way authorities and environmental agencies monitor and manage water quality. In particular, we aim to empower the "
            "Laguna Lake Development Authority (LLDA), which has been grappling with the increasing prevalence of harmful algal blooms "
            "(HABs) in Laguna Lake. These blooms pose a serious threat to water quality, marine life, and local communities by depleting "
            "oxygen in the water and releasing toxins that can harm both aquatic species and human health.\n\n"
            
            "At the core of our system is the use of Long Short-Term Memory (LSTM), a powerful form of Recurrent Neural Networks (RNNs), "
            "which can analyze and predict complex, time-series data. By using historical and real-time data on water quality parameters "
            "such as temperature, nutrient levels (like nitrogen and phosphorus), and meteorological factors, we aim to provide accurate "
            "predictions of when and where algal blooms are likely to occur. This predictive capability will help the LLDA and local environmental "
            "agencies make proactive, data-driven decisions to mitigate the effects of these blooms before they become catastrophic.\n\n"
            
            "Our team’s overarching goal is not only to build a technologically advanced system for algal bloom prediction but also to bridge "
            "the gap between computer science and environmental sustainability. We believe that by leveraging the power of machine learning and "
            "data analytics, we can help communities make informed decisions that contribute to the preservation of Laguna Lake’s ecological balance "
            "and, by extension, the health of similar aquatic ecosystems around the world.\n\n"
            
            "We envision BloomSentry as a tool that will enable a more responsive, efficient, and scientifically informed approach to environmental "
            "management. It is our belief that the future of environmental sustainability lies in the integration of advanced technology with real-world "
            "environmental efforts, and we are committed to being part of that change.\n\n"
            
            "Through this project, we hope to make a meaningful contribution to both data science and environmental conservation, showing how innovative "
            "technologies can be applied to create a greener, healthier future for all."
        )


        self.project_label = ctk.CTkLabel(
            self.content_frame,
            text=self.project_info,
            wraplength=1000,
            justify="left",
            font=("Roboto", 12),
            anchor="center"
        )
        self.project_label.grid(row=0, column=0, rowspan=5, columnspan=2, pady=10, padx=20, sticky="ew")

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")
