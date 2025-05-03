import customtkinter as ctk
import tkinter as tk

from sidebar import Sidebar
from dashboard import DashboardPage
from about import AboutPage
from inputData import InputDataPage
from waterQualityReport import WaterQualityReport
from prediction import PredictionPage
from settings import SettingsPage
from audit import get_audit_logger
from icons import IconManager
from globals import current_user_key


class Main(ctk.CTk):
    def __init__(self, current_user_key, user_type="regular"):
        super().__init__()

        # Set appearance mode and color theme
        ctk.set_appearance_mode("light")  # Options: "light", "dark", "system"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        self.current_user_key = current_user_key
        self.user_type = user_type

        # Initialize audit logger
        self.audit_logger = get_audit_logger()
        # Log user login/session start
        self.audit_logger.log_login(self.current_user_key, self.user_type)
        
        print(f"User: {current_user_key}, Type: {user_type}")
        self.title("Bloom Sentry")

        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Set window to full screen dimensions
        self.geometry(f"{screen_width}x{screen_height}")
        
        # Configure main window to expand with screen size
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)  # Column 1 (main content) should expand

        # CustomTkinter uses scaling differently
        ctk.set_window_scaling(1.2)

        # Instantiate Icon Manager
        self.icon_manager = IconManager()
        
        # Initialize sidebar with fixed width
        self.sidebar_width = 200  # Using the max_width as fixed width
        
        # Mainframe (adjusts based on screen size)
        self.mainFrame = ctk.CTkFrame(
            self, 
            width=(screen_width - self.sidebar_width), 
            height=screen_height,
            fg_color="#FFFFFF"
        )
        self.mainFrame.grid(row=0, column=1, sticky="nsew")
        self.mainFrame.rowconfigure(0, weight=1)
        self.mainFrame.columnconfigure(0, weight=1)

        # Create sidebar
        self.sidebar = Sidebar(self, self, self.icon_manager, self.mainFrame, user_type=self.user_type)

        # Initialize pages
        self.dashboard = DashboardPage(self.mainFrame)
        self.about = AboutPage(self.mainFrame)
        self.input = InputDataPage(self.mainFrame, current_username=current_user_key, user_type=user_type)
        self.report = WaterQualityReport(self.mainFrame)
        self.predict = PredictionPage(self.mainFrame)
        self.settings = SettingsPage(self.mainFrame, current_user_key, user_type)

        # Make application respond to window resizing
        self.bind("<Configure>", self.on_resize)
        
        # Register the WM_DELETE_WINDOW protocol to handle app closure
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Show dashboard by default
        self.call_page(None, self.dashboard.show)

    def on_resize(self, event):
        """Handle window resize events to adjust layout"""
        # Only process events from the main window
        if event.widget == self:
            # Update mainFrame size based on new window dimensions
            new_width = event.width - self.sidebar_width
            new_height = event.height
            
            # Resize mainFrame to fill available space
            self.mainFrame.configure(width=new_width, height=new_height)

    def forget_page(self):
        for frame in self.mainFrame.winfo_children():
            frame.grid_forget()

    def call_page(self, btn, page):
        self.sidebar.reset_indicator()
        if btn is not None:
            # For CustomTkinter, we use different methods to indicate selection
            btn.configure(border_width=2, border_color="#F1F1F1")
        self.forget_page()
        page()
    
    def on_closing(self):
        """Handle application closing."""
        # Log user logout
        self.audit_logger.log_logout(self.current_user_key, self.user_type)
        # Destroy the application
        self.destroy()


if __name__ == "__main__":
    app = Main(current_user_key)
    # Start in full screen mode - Windows only
    app.mainloop()