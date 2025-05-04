import customtkinter as ctk
from PIL import Image, ImageDraw


class AboutPage(ctk.CTkFrame):
    def __init__(self, parent, bg_color=None):
        # Define color scheme based on #1f6aa5
        self.colors = {
            "primary": "#1f6aa5",          # Base blue color
            "primary_light": "#2d7dbe",    # Lighter blue for hover effects
            "primary_dark": "#18558a",     # Darker blue for accent elements
            "secondary": "#f0f7ff",        # Very light blue for backgrounds
            "accent": "#ff8c42",           # Orange complementary color
            "text_dark": "#2c3e50",        # Dark blue-gray for main text
            "text_light": "#ffffff",       # White for text on dark backgrounds
            "card_bg": "#ffffff",          # White for card backgrounds
            "border": "#d1e3f6",           # Light blue for borders
            "dev_cards": [                 # Developer card background colors (subtle variations)
                "#ebf5ff",                 # Very light blue
                "#e6f2ff",                 # Slightly different light blue
                "#e1eeff",                 # Another subtle light blue
                "#dcebff"                  # Slightly darker light blue
            ],
            "accent_gradients": [          # Accent colors for team member cards
                "#1f6aa5",                 # Primary blue
                "#3498db",                 # Lighter blue
                "#2980b9",                 # Medium blue
                "#206694"                  # Darker blue
            ]
        }
        
        super().__init__(parent, fg_color=bg_color or self.colors["secondary"])
        self.parent = parent
        self.propagate(False)
        self.create_widgets()
        self.grid_columnconfigure(0, weight=1)
        self.propagate(False)

    def create_widgets(self):
        aboutlb = ctk.CTkLabel(
            self, text="ABOUT US", 
            justify="left", anchor="w",
            font=("Segoe UI", 25, "bold"),
            text_color=self.colors["primary_dark"]
        )
        aboutlb.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")

        # Create scrollable container with themed background
        self.scrollable_container = ctk.CTkScrollableFrame(
            self, 
            fg_color=self.colors["secondary"],
            scrollbar_fg_color=self.colors["secondary"],
            scrollbar_button_color=self.colors["primary"],
            scrollbar_button_hover_color=self.colors["primary_light"]
        )
        self.scrollable_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 20))
        self.scrollable_container.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        # Content frame inside scrollable container with themed background
        self.content_frame = ctk.CTkFrame(self.scrollable_container, fg_color="transparent")
        self.content_frame.grid(row=0, column=0, sticky="nsew")
        self.content_frame.columnconfigure(0, weight=1)

        # Project Description - NOW AT THE TOP
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
            
            "Our team's overarching goal is not only to build a technologically advanced system for algal bloom prediction but also to bridge "
            "the gap between computer science and environmental sustainability. We believe that by leveraging the power of machine learning and "
            "data analytics, we can help communities make informed decisions that contribute to the preservation of Laguna Lake's ecological balance "
            "and, by extension, the health of similar aquatic ecosystems around the world.\n\n"
            
            "We envision BloomSentry as a tool that will enable a more responsive, efficient, and scientifically informed approach to environmental "
            "management. It is our belief that the future of environmental sustainability lies in the integration of advanced technology with real-world "
            "environmental efforts, and we are committed to being part of that change.\n\n"
            
            "Through this project, we hope to make a meaningful contribution to both data science and environmental conservation, showing how innovative "
            "technologies can be applied to create a greener, healthier future for all."
        )

        # Project Info Section with distinct styling
        project_section = ctk.CTkFrame(
            self.content_frame,
            corner_radius=15,
            fg_color=self.colors["card_bg"],
            border_width=1,
            border_color=self.colors["border"]
        )
        project_section.grid(row=0, column=0, pady=(10, 30), padx=20, sticky="ew")
        project_section.columnconfigure(0, weight=1)
        
        # Add blue top bar to project section
        top_bar = ctk.CTkFrame(
            project_section,
            height=8,
            corner_radius=0,
            fg_color=self.colors["primary"]
        )
        top_bar.grid(row=0, column=0, sticky="ew")
        
        # Project Section Title
        project_title = ctk.CTkLabel(
            project_section,
            text="ABOUT OUR PROJECT",
            font=("Segoe UI", 16, "bold"),
            anchor="w",
            text_color=self.colors["primary"]
        )
        project_title.grid(row=1, column=0, padx=20, pady=(15, 5), sticky="w")
        
        # Project Description
        self.project_label = ctk.CTkLabel(
            project_section,
            text=self.project_info,
            wraplength=950,
            justify="left",
            font=("Segoe UI", 12),
            anchor="w",
            text_color=self.colors["text_dark"]
        )
        self.project_label.grid(row=2, column=0, pady=(5, 20), padx=20, sticky="ew")

        # Team section container with themed styling
        team_section = ctk.CTkFrame(
            self.content_frame,
            corner_radius=15,
            fg_color=self.colors["card_bg"],
            border_width=1,
            border_color=self.colors["border"]
        )
        team_section.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        team_section.columnconfigure(0, weight=1)
        
        # Add blue top bar to team section
        team_top_bar = ctk.CTkFrame(
            team_section,
            height=8,
            corner_radius=0,
            fg_color=self.colors["primary"]
        )
        team_top_bar.grid(row=0, column=0, sticky="ew")
        
        # Developer section title
        team_title = ctk.CTkLabel(
            team_section, 
            text="MEET THE TEAM", 
            font=("Segoe UI", 18, "bold"),
            anchor="w",
            text_color=self.colors["primary"]
        )
        team_title.grid(row=1, column=0, padx=20, pady=(15, 10), sticky="w")
        
        def make_rounded_square_image(image_path, size=(100, 100), corner_radius=15):
            """Create a rounded square image using PIL and CTkImage"""
            img = Image.open(image_path).resize(size, Image.LANCZOS)
            
            # Create a mask with rounded corners
            mask = Image.new("L", size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle([(0, 0), (size[0], size[1])], corner_radius, fill=255)

            img_rgba = img.convert("RGBA")
            rounded_img = Image.new("RGBA", size, (0, 0, 0, 0))
            rounded_img.paste(img_rgba, (0, 0), mask)

            return ctk.CTkImage(light_image=rounded_img, dark_image=rounded_img, size=size)

        # Developer Images
        self.P1 = make_rounded_square_image("DevPics/Benj.jpg", size=(120, 120))
        self.P2 = make_rounded_square_image("DevPics/Matt.jpg", size=(120, 120))
        self.P3 = make_rounded_square_image("DevPics/Rafi.jpg", size=(120, 120))
        self.P4 = make_rounded_square_image("DevPics/Beau.jpg", size=(120, 120))

        # Developer Info with descriptions
        devs = [
            (self.P1, "Franz Benjamin Africano", "Team Lead & Machine Learning Specialist - Responsible for developing and fine-tuning the LSTM model for algal bloom prediction."),
            (self.P2, "Matt Terrence Rias", "Data Scientist & Backend Developer - Focused on data preprocessing and integration of water quality parameters into the prediction system."),
            (self.P3, "Mohammad Rafi Saiyari", "UI/UX Designer & Frontend Developer - Created the user interface and data visualization components for BloomSentry."),
            (self.P4, "Beau Lawyjet Sison", "Environmental Data Analyst - Specialized in interpreting environmental data and ensuring the system's accuracy for Laguna Lake's specific conditions.")
        ]

        # Create developer cards with horizontal layout (photo beside text)
        for idx, (profile_pic, name, description) in enumerate(devs):
            card = ctk.CTkFrame(
                team_section, 
                corner_radius=15, 
                fg_color="#ffffff",
                border_width=1,
                border_color="#d0d0d0"  # Light border
            )
            card.grid(
                row=idx+1, column=0, padx=20, pady=10, sticky="ew"
            )
            card.columnconfigure(1, weight=1)  # Make text column expandable
            
            # Create colored accent on the left side of each card
            accent_colors = ["#1f6aa5", "#1f6aa5", "#1f6aa5", "#1f6aa5"]  # Different color for each dev
            accent = ctk.CTkFrame(
                card, 
                width=8, 
                corner_radius=0,
                fg_color=accent_colors[idx % len(accent_colors)]
            )
            accent.grid(row=0, column=0, rowspan=2, sticky="ns")
            
            # Photo beside the accent
            img_label = ctk.CTkLabel(card, image=profile_pic, text="")
            img_label.grid(row=0, column=1, rowspan=2, padx=(15, 15), pady=20)
            
            # Name to the right of the photo
            name_label = ctk.CTkLabel(
                card, text=name, font=("Roboto", 14, "bold"), anchor="w"
            )
            name_label.grid(row=0, column=2, padx=(0, 20), pady=(20, 5), sticky="w")
            
            # Description below the name
            desc_label = ctk.CTkLabel(
                card, text=description, font=("Roboto", 12), anchor="w",
                wraplength=750, justify="left"
            )
            desc_label.grid(row=1, column=2, padx=(0, 20), pady=(0, 20), sticky="w")

    def show(self):
        # Ensure the main frame takes up all available space
        self.grid(row=0, column=0, sticky="nsew")
        
        # Make sure the scrollable container expands to fill the available space
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Add a subtle blue gradient background to the main frame
        self.configure(fg_color=self.colors["secondary"])