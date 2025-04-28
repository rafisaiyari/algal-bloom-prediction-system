import customtkinter as ctk
from PIL import Image

# Initial width settings
min_w = 85
cur_width = min_w
expanded = False
is_hovering = False


class Sidebar(ctk.CTkFrame):
    def __init__(self, parent, controller, icon_manager, mainFrame, user_type="regular"):
        super().__init__(parent, fg_color="#1d97bd", width=85, height=parent.winfo_height())
        self.user_type = user_type
        self.parent = parent
        self.controller = controller
        self.mainFrame = mainFrame
        self.propagate(False)

        # Initialize dimensions
        self.min_w = 85
        self.max_w = 200
        self.cur_width = self.min_w
        self.expanded = False
        self.is_hovering = False

        # Load icons using CTkImage instead of ImageTk.PhotoImage
        self.load_icons()

        self.grid(row=0, column=0, sticky="ns")

        # Create sidebar buttons
        self.create_buttons()

        # Set up hover events
        self.setup_bindings()

    def load_icons(self):
        """Load icons using CTkImage for proper high DPI scaling"""
        # Define image paths
        image_paths = {
            "AboutIcon": "Icons/AppLogo.png",
            "AppIcon": "Icons/AppIcon.png",
            "AppLogo": "Icons/AppLogo.png",
            "DBIcon": "Icons/DBIcon.png",
            "IPIcon": "Icons/IPIcon.png",
            "WQRIcon": "Icons/WQRIcon.png",
            "PTIcon": "Icons/PTIcon.png",
            "SIcon": "Icons/SIcon.png"
        }

        # Define image sizes
        image_sizes = {
            "AboutIcon": (200, 65),
            "AppIcon": (35, 35),
            "AppLogo": (120, 40),
            "DBIcon": (25, 25),
            "IPIcon": (25, 25),
            "WQRIcon": (25, 25),
            "PTIcon": (25, 25),
            "SIcon": (25, 25)
        }

        # Create CTkImage objects for each icon
        self.icons = {}
        for name, path in image_paths.items():
            try:
                size = image_sizes.get(name, (25, 25))
                # Light and dark mode can use the same image, or you can specify different ones
                self.icons[name] = ctk.CTkImage(
                    light_image=Image.open(path),
                    dark_image=Image.open(path),
                    size=size
                )
            except Exception as e:
                print(f"Error loading icon '{name}' from '{path}': {e}")

    def create_buttons(self):
        """Create all sidebar buttons"""
        # About button
        self.btn1 = ctk.CTkButton(
            self,
            image=self.icons["AppIcon"],
            text="",
            cursor="hand2",
            anchor="center",
            fg_color="#F1F1F1",
            hover_color="#E0E0E0",
            corner_radius=6,
            command=lambda: self.controller.call_page(self.btn1, self.controller.about.show)
        )

        # Dashboard button
        self.btn2 = ctk.CTkButton(
            self,
            image=self.icons["DBIcon"],
            text="",
            cursor="hand2",
            anchor="center",
            fg_color = "#1d97bd",
            hover_color = "#1a86a8",
            corner_radius = 6,
            command = lambda: self.controller.call_page(self.btn2, self.controller.dashboard.show)
        )

        # Input Data button
        self.btn3 = ctk.CTkButton(
            self,
            image=self.icons["IPIcon"],
            text="",
            cursor="hand2",
            anchor="center",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            command=lambda: self.controller.call_page(self.btn3, self.controller.input.show)
        )

        # Disable button if user is "regular"
        if self.user_type == "regular":
            self.btn3.configure(state="disabled")

        # Water Quality Report button
        self.btn4 = ctk.CTkButton(
            self,
            image=self.icons["WQRIcon"],
            text="",
            cursor="hand2",
            anchor="center",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            command=lambda: self.controller.call_page(self.btn4, self.controller.report.show)
        )

        # Prediction Tool button
        self.btn5 = ctk.CTkButton(
            self,
            image=self.icons["PTIcon"],
            text="",
            cursor="hand2",
            anchor="center",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            command=lambda: self.controller.call_page(self.btn5, self.controller.predict.show)
        )

        # Settings button
        self.btn6 = ctk.CTkButton(
            self,
            image=self.icons["SIcon"],
            text="",
            cursor="hand2",
            anchor="center",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            command=lambda: self.controller.call_page(self.btn6, self.controller.settings.show)
        )

        # Pack buttons in the sidebar
        self.btn1.pack(fill="x", padx=20, pady=20)
        self.btn2.pack(fill="x", padx=20, pady=15)
        self.btn3.pack(fill="x", padx=20, pady=15)
        self.btn4.pack(fill="x", padx=20, pady=15)
        self.btn5.pack(fill="x", padx=20, pady=15)
        self.btn6.pack(side="bottom", fill="x", padx=20, pady=25)

    def setup_bindings(self):
        """Set up mouse hover bindings for the sidebar and buttons"""
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

        for btn in [self.btn1, self.btn2, self.btn3, self.btn4, self.btn5, self.btn6]:
            btn.bind('<Enter>', self.on_enter)
            btn.bind('<Leave>', self.on_leave)

    def reset_indicator(self):
        """Reset all button indicators to default state"""
        self.btn1.configure(fg_color="#F1F1F1", border_width=0)
        self.btn2.configure(fg_color="#1d97bd", border_width=0)
        self.btn3.configure(fg_color="#1d97bd", border_width=0)
        self.btn4.configure(fg_color="#1d97bd", border_width=0)
        self.btn5.configure(fg_color="#1d97bd", border_width=0)
        self.btn6.configure(fg_color="#1d97bd", border_width=0)

    def expand(self):
        """Expand the sidebar animation"""
        self.cur_width += 5
        self.rep = self.parent.after(5, self.expand)
        self.configure(width=self.cur_width)

        if self.cur_width >= self.max_w:
            self.expanded = True
            self.parent.after_cancel(self.rep)
            self.cur_width = self.max_w
            self.fill()

    def contract(self):
        """Contract the sidebar animation"""
        self.cur_width -= 5
        self.rep = self.parent.after(5, self.contract)
        self.configure(width=self.cur_width)

        if self.cur_width <= self.min_w:
            self.expanded = False
            self.parent.after_cancel(self.rep)
            self.cur_width = self.min_w
            self.fill()

    def fill(self):
        """Update button appearance based on sidebar state"""
        if self.expanded:
            # Expanded state: Show text and icons
            self.btn1.configure(text="", image=self.icons["AppLogo"], width=140, height=35, fg_color="#FFFFFF")
            self.btn2.configure(text=" DASHBOARD", image=self.icons["DBIcon"], compound="left",
                                text_color="#f1f1f1", width=140, fg_color="#1d97bd", anchor="w")
            self.btn3.configure(text=" INPUT DATA", image=self.icons["IPIcon"], compound="left",
                                text_color="#f1f1f1", width=140, fg_color="#1d97bd", anchor="w")
            self.btn4.configure(text=" WATER QUALITY\nREPORT", image=self.icons["WQRIcon"], compound="left",
                                text_color="#f1f1f1", width=140, fg_color="#1d97bd", anchor="w")
            self.btn5.configure(text=" PREDICTION TOOL", image=self.icons["PTIcon"], compound="left",
                                text_color="#f1f1f1", width=140, fg_color="#1d97bd", anchor="w")
            self.btn6.configure(text=" SETTINGS", image=self.icons["SIcon"], compound="left",
                                text_color="#f1f1f1", width=140, fg_color="#1d97bd", anchor="w")
        else:
            # Collapsed state: Show only icons
            self.btn1.configure(text="", image=self.icons["AppIcon"], width=35, height=35)
            self.btn2.configure(text="", image=self.icons["DBIcon"], width=35, anchor="center")
            self.btn3.configure(text="", image=self.icons["IPIcon"], width=35, anchor="center")
            self.btn4.configure(text="", image=self.icons["WQRIcon"], width=35, anchor="center")
            self.btn5.configure(text="", image=self.icons["PTIcon"], width=35, anchor="center")
            self.btn6.configure(text="", image=self.icons["SIcon"], width=35, anchor="center")

    def on_enter(self, event):
        """Handle mouse enter events"""
        self.is_hovering = True
        if not self.expanded:
            self.expand()

    def on_leave(self, event):
        """Handle mouse leave events"""
        self.is_hovering = False
        self.parent.after(100, self.check_and_contract)

    def check_and_contract(self):
        """Check if mouse is still outside and contract sidebar if needed"""
        if not self.is_hovering:
            self.contract()