import customtkinter as ctk
from PIL import Image, ImageDraw


class AboutPage(ctk.CTkFrame):
    def __init__(self, parent, bg_color=None):
        # Define color scheme based on sidebar.py colors
        self.colors = {
            "primary": "#1f6aa5",          # Base blue color from sidebar
            "primary_light": "#17537f",    # Hover color from sidebar
            "primary_dark": "#144463",     # Pressed color from sidebar
            "secondary": "#f0f7ff",        # Very light blue for backgrounds
            "accent": "#ff8c42",           # Orange complementary color
            "text_dark": "#2c3e50",        # Primary Text - For main body text, labels, and headers
            "text_secondary": "#5d7285",   # Secondary Text - For supporting text, captions, and placeholders
            "text_light": "#f1f1f1",       # Text color from sidebar
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
                "#17537f",                 # Hover color from sidebar
                "#144463",                 # Pressed color from sidebar
                "#206694"                  # Darker blue
            ]
        }
        
        super().__init__(parent, fg_color=bg_color or self.colors["secondary"])
        self.parent = parent
        self.propagate(False)
        
        # Configure main frame to expand with window
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Create a canvas for scrolling - this approach matches inputData.py
        self.canvas = ctk.CTkCanvas(self, highlightthickness=0, bg=self._apply_appearance_mode(self.colors["secondary"]))
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar
        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Create a frame inside the canvas to hold all content
        self.content_frame = ctk.CTkFrame(self.canvas, fg_color="transparent")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        
        # Configure content frame
        self.content_frame.columnconfigure(0, weight=1)
        
        # Create widgets inside the content frame
        self.create_widgets()
        
        # Configure events for proper scrolling and resizing
        self.content_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Bind mousewheel events for scrolling
        self.bind_mousewheel()

    def on_frame_configure(self, event):
        """Update the scroll region based on the content frame size"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Adjust the width of the canvas window when canvas is resized"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def bind_mousewheel(self):
        """Bind mousewheel to scroll the canvas - matching inputData.py approach"""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind for Windows and macOS
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Bind for Linux
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))

    def create_widgets(self):
        aboutlb = ctk.CTkLabel(
            self.content_frame, text="ABOUT US: THE DEVELOPERS", 
            justify="left", anchor="w",
            font=("Segoe UI", 25, "bold"),
            text_color=self.colors["text_dark"]  # Use primary text color for main header
        )
        aboutlb.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")

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
            
            "At the core of our system is the use of Gradient Boosted Regression (GBR), a powerful ensemble machine learning technique, "  
            "which can analyze and predict complex, time-series data. By using historical and real-time data on water quality parameters "  
            "such as temperature, nutrient levels, and meteorological factors, we aim to provide accurate predictions of when and where"  
            "algal blooms are likely to occur. This predictive capability will help the LLDA and local environmental agencies make proactive,"  
            "data-driven decisions to mitigate the effects of these blooms before they become catastrophic.\n\n"

            
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

        # Project Info Section with distinct styling - maximize the frame
        project_section = ctk.CTkFrame(
            self.content_frame,
            corner_radius=15,
            fg_color=self.colors["card_bg"],
            border_width=1,
            border_color=self.colors["border"]
        )
        project_section.grid(row=1, column=0, pady=(10, 30), padx=20, sticky="ew")
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
            text_color=self.colors["primary"]  # Keep primary color for section titles for emphasis
        )
        project_title.grid(row=1, column=0, padx=20, pady=(15, 5), sticky="w")
        
        # Project Description - maximize the text container
        self.project_label = ctk.CTkLabel(
            project_section,
            text=self.project_info,
            wraplength=1200,  # Increased wraplength to maximize space usage
            justify="left",
            font=("Segoe UI", 12),
            anchor="w",
            text_color=self.colors["text_dark"]  # Main body text uses primary text color
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
        team_section.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
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
            text_color=self.colors["primary"]  # Keep primary color for section titles for emphasis
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
            (self.P1, "Franz Benjamin Africano", "UI/UX Designer & Frontend Developer - Created the user interface and data visualization components for BloomSentry."),
            (self.P2, "Matt Terrence Rias", "Data Scientist & Backend Developer - Focused on data preprocessing and integration of water quality parameters into the prediction system."),
            (self.P3, "Mohammad Rafi Saiyari", "Team Lead & Machine Learning Specialist - Responsible for developing and fine-tuning the LSTM model for algal bloom prediction."),
            (self.P4, "Beau Lawyjet Sison", "Environmental Data Analyst - Specialized in interpreting environmental data and ensuring the system's accuracy for Laguna Lake's specific conditions.")
        ]

        # Modified developer cards with pictures aligned to the leftmost side
        for idx, (profile_pic, name, description) in enumerate(devs):
            card = ctk.CTkFrame(
                team_section, 
                corner_radius=15, 
                fg_color="#ffffff",
                border_width=1,
                border_color="#d0d0d0"  # Light border
            )
            card.grid(row=idx+2, column=0, padx=20, pady=10, sticky="ew")
            card.columnconfigure(2, weight=1)  # Make text column expandable
            
            # Create colored accent on the left side of each card - using primary color consistently
            accent = ctk.CTkFrame(
                card, 
                width=8, 
                corner_radius=0,
                fg_color=self.colors["primary"]  # Use the primary color for consistency
            )
            accent.grid(row=0, column=0, sticky="ns")
            
            # Photo placed immediately after the accent (leftmost position)
            img_label = ctk.CTkLabel(card, image=profile_pic, text="")
            img_label.grid(row=0, column=1, padx=(10, 15), pady=15, sticky="w")
            
            # Text container - positioned to align with the image
            text_container = ctk.CTkFrame(
                card,
                fg_color="transparent"  # Make it transparent to blend with card background
            )
            text_container.grid(row=0, column=2, padx=(0, 20), pady=(15, 15), sticky="nw")
            
            # Name aligned with the top of the image
            name_label = ctk.CTkLabel(
                text_container, 
                text=name, 
                font=("Segoe UI", 24, "bold"), 
                anchor="w",
                justify="left"
            )
            name_label.grid(row=0, column=0, pady=(30,10), sticky="nw")
            
            # Description directly below the name
            desc_label = ctk.CTkLabel(
                text_container, 
                text=description, 
                font=("Segoe UI", 12), 
                anchor="w",
                wraplength=780,
                justify="left",
                text_color=self.colors["text_secondary"]  # Use secondary text color for descriptions
            )
            desc_label.grid(row=1, column=0, pady=(8, 0), sticky="nw")

    def show(self):
        # Ensure the main frame takes up all available space
        self.grid(row=0, column=0, sticky="nsew")
        
        # Add a subtle blue gradient background to the main frame
        self.configure(fg_color=self.colors["secondary"])
        
        # Unbind any previous mousewheel bindings before rebinding to prevent duplicates
        try:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        except:
            pass
        
        # Rebind the mousewheel events
        self.bind_mousewheel()