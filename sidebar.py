import customtkinter as ctk
from PIL import Image


class Sidebar(ctk.CTkFrame):
    def __init__(self, parent, controller, icon_manager, mainFrame, user_type="regular"):
        # Fixed width of 200px (previously max_w in the animated version)
        self.fixed_width = 200
        
        super().__init__(parent, fg_color="#1d97bd", width=self.fixed_width, height=parent.winfo_height())
        self.user_type = user_type
        self.parent = parent
        self.controller = controller
        self.mainFrame = mainFrame
        self.propagate(False)

        # Load icons using icon_manager
        self.icon_manager = icon_manager
        self.AboutIcon = self.icon_manager.get_icon("AboutIcon")
        self.AppIcon = self.icon_manager.get_icon("AppIcon")
        self.DBIcon = self.icon_manager.get_icon("DBIcon")
        self.IPIcon = self.icon_manager.get_icon("IPIcon")
        self.WQRIcon = self.icon_manager.get_icon("WQRIcon")
        self.PTIcon = self.icon_manager.get_icon("PTIcon")
        self.SIcon = self.icon_manager.get_icon("SIcon")

        self.grid(row=0, column=0, sticky="ns")

        # Create sidebar buttons with fixed width
        self.create_buttons()

    def create_buttons(self):
        """Create all sidebar buttons with fixed width"""
        # About button - special handling with frame
        self.btn1_frame = ctk.CTkFrame(
            self,
            fg_color="#F1F1F1",
            width=160,
            height=35,
            corner_radius=6,
            border_width=0
        )
        # More precise padding to prevent the blue line
        self.btn1_frame.pack(fill="x", padx=20, pady=(20, 20))
        self.btn1_frame.pack_propagate(False)

        # Create a merged button that contains both icon and text
        self.btn1_merged = ctk.CTkButton(
            self.btn1_frame,
            image=self.AppIcon,
            text="BloomSentry",  # Always show text since sidebar is fixed width
            text_color="#1d97bd",
            font=("Arial", 12, "bold"),
            cursor="hand2",
            fg_color="#F1F1F1",
            hover_color="#E0E0E0",
            corner_radius=6,
            width=160,  # Fixed width
            height=35,
            border_width=0,
            anchor="w",
            compound="left",
            command=lambda: self.controller.call_page(self.btn1_frame, self.controller.about.show)
        )
        self.btn1_merged.pack(fill="both", expand=True)

        # Add a tiny white "cover" frame that will sit at the bottom edge of the button frame
        # to prevent the blue line from showing through
        self.btn1_cover = ctk.CTkFrame(
            self.btn1_frame,
            fg_color="#F1F1F1",
            height=1,
            corner_radius=0,
            border_width=0
        )
        self.btn1_cover.pack(side="bottom", fill="x", pady=(0, 0))

        # Store a reference to the frame for the controller to use
        self.btn1 = self.btn1_frame

        # Dashboard button
        self.btn2 = ctk.CTkButton(
            self,
            image=self.DBIcon,
            text=" DASHBOARD",  # Always show text
            text_color="#f1f1f1",
            cursor="hand2",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            width=160,  # Fixed width
            height=35,
            anchor="w",
            compound="left",
            command=lambda: self.controller.call_page(self.btn2, self.controller.dashboard.show)
        )

        # Input Data button
        self.btn3 = ctk.CTkButton(
            self,
            image=self.IPIcon,
            text=" INPUT DATA",  # Always show text
            text_color="#f1f1f1",
            cursor="hand2",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            width=160,  # Fixed width
            height=35,
            anchor="w",
            compound="left",
            command=lambda: self.controller.call_page(self.btn3, self.controller.input.show)
        )

        # Disable button if user is "regular"
        if self.user_type == "regular":
            self.btn3.configure(state="disabled")

        # Water Quality Report button
        self.btn4 = ctk.CTkButton(
            self,
            image=self.WQRIcon,
            text=" WATER QUALITY\nREPORT",  # Always show text
            text_color="#f1f1f1",
            cursor="hand2",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            width=160,  # Fixed width
            height=35,
            anchor="w",
            compound="left",
            command=lambda: self.controller.call_page(self.btn4, self.controller.report.show)
        )

        # Prediction Tool button
        self.btn5 = ctk.CTkButton(
            self,
            image=self.PTIcon,
            text=" PREDICTION TOOL",  # Always show text
            text_color="#f1f1f1",
            cursor="hand2",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            width=160,  # Fixed width
            height=35,
            anchor="w",
            compound="left",
            command=lambda: self.controller.call_page(self.btn5, self.controller.predict.show)
        )

        # Settings button
        self.btn6 = ctk.CTkButton(
            self,
            image=self.SIcon,
            text=" SETTINGS",  # Always show text
            text_color="#f1f1f1",
            cursor="hand2",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            width=160,  # Fixed width
            height=35,
            anchor="w",
            compound="left",
            command=lambda: self.controller.call_page(self.btn6, self.controller.settings.show)
        )

        # Place buttons
        self.btn2.pack(fill="x", padx=20, pady=15)
        self.btn3.pack(fill="x", padx=20, pady=15)
        self.btn4.pack(fill="x", padx=20, pady=15)
        self.btn5.pack(fill="x", padx=20, pady=15)
        self.btn6.pack(side="bottom", fill="x", padx=20, pady=25)

    def reset_indicator(self):
        """Reset all button indicators to default state"""
        self.btn1_frame.configure(fg_color="#F1F1F1")
        self.btn1_merged.configure(fg_color="#F1F1F1", border_width=0)
        self.btn2.configure(fg_color="#1d97bd", border_width=0)
        self.btn3.configure(fg_color="#1d97bd", border_width=0)
        self.btn4.configure(fg_color="#1d97bd", border_width=0)
        self.btn5.configure(fg_color="#1d97bd", border_width=0)
        self.btn6.configure(fg_color="#1d97bd", border_width=0)