import customtkinter as ctk

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
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

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

        # Set initial window size to 90% of screen size
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)

        # Calculate position for center of screen
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        # Set window size and position
        self.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # After setting geometry, try to maximize based on OS
        self.after(100, self.maximize_window)

        self.minsize(800, 600)
        self.propagate(False)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # CustomTkinter uses scaling differently
        ctk.set_window_scaling(1.2)

        # Instantiate Icon Manager
        self.icon_manager = IconManager()
        
        # Initialize sidebar with fixed width
        self.sidebar_width = 200  # Using the max_width as fixed width
        
        # Mainframe (adjusts based on screen size)
        self.mainFrame = ctk.CTkFrame(
            self, 
            width=(self.winfo_width() - 100), 
            height=self.winfo_height(),
            fg_color="#FFFFFF"
        )
        self.mainFrame.grid(row=0, column=1, sticky="nsew")
        self.mainFrame.rowconfigure(0, weight=1)
        self.mainFrame.columnconfigure(0, weight=1)

        ctk.set_window_scaling(1.2)
        self.icon_manager = IconManager()

        # Create sidebar
        self.sidebar = Sidebar(self, self, self.icon_manager, self.mainFrame, user_type=self.user_type)

        # Initialize pages
        self.dashboard = DashboardPage(self.mainFrame)
        self.about = AboutPage(self.mainFrame)
        self.input = InputDataPage(self.mainFrame, current_username=current_user_key, user_type=user_type)
        self.report = WaterQualityReport(self.mainFrame)
        self.predict = PredictionPage(self.mainFrame)
        self.settings = SettingsPage(self.mainFrame, current_user_key, user_type)

        # Show dashboard by default
        self.call_page(None, self.dashboard.show)

    def maximize_window(self):
        """Try multiple approaches to maximize the window"""
        try:
            # Windows method
            self.state('zoomed')
        except Exception as e:
            try:
                # macOS method
                self.attributes('-zoomed', True)
            except Exception:
                # Linux/other fallback
                screen_width = self.winfo_screenwidth()
                screen_height = self.winfo_screenheight()
                self.geometry(f"{screen_width}x{screen_height}+0+0")
    
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

        current_active = None
        for frame in self.mainFrame.winfo_children():
            if frame.winfo_viewable():
                current_active = frame
                break

        # Clean up if it's the report page
        if current_active is self.report and hasattr(self.report, 'cleanup'):
            print("Cleaning up Water Quality Report before switching pages")
            self.report.cleanup()
        self.forget_page()
        page()

    
    def cleanup_current_page(self):
        """Clean up resources for the current page before switching"""
        # Check which page is currently visible and clean it up
        for frame in self.mainFrame.winfo_children():
            if frame.winfo_viewable():
                if isinstance(frame, WaterQualityReport):
                    print("Cleaning up Water Quality Report")
                    frame.cleanup()
                    break

    def update_water_quality_report(self):
        """Refresh the water quality report with new data"""
        if hasattr(self, 'report'):
            self.report.cleanup()
            # Force reload data (if needed)
            self.report._data_cache['initialized'] = False
            # Show the report again to reload
            self.call_page(None, self.report.show)

    
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