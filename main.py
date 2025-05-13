import customtkinter as ctk
from sidebar import Sidebar
from dashboard import DashboardPage
from about import AboutPage
from inputData import InputDataPage
from waterQualityReport import WaterQualityReport
from prediction import PredictionPage
from settings import SettingsPage
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
        print(f"User: {current_user_key}, Type: {user_type}")
        self.title("Bloom Sentry")

        # Get screen dimensions for proper scaling
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

        # Mainframe
        self.mainFrame = ctk.CTkFrame(self, width=(self.winfo_width() - 100), height=self.winfo_height(),
                                      fg_color="#FFFFFF")
        self.mainFrame.grid(row=0, column=1, sticky="nsew")
        self.mainFrame.rowconfigure(0, weight=1)
        self.mainFrame.columnconfigure(0, weight=1)

        # CustomTkinter uses scaling differently
        ctk.set_window_scaling(1.2)

        # Instantiate Sidebar, and Icon Manager
        self.icon_manager = IconManager()
        self.sidebar = Sidebar(self, self, self.icon_manager, self.mainFrame, user_type=self.user_type)

        self.dashboard = DashboardPage(self.mainFrame)
        self.about = AboutPage(self.mainFrame)
        self.input = InputDataPage(self.mainFrame)
        self.report = WaterQualityReport(self.mainFrame)
        self.predict = PredictionPage(self.mainFrame)
        self.settings = SettingsPage(self.mainFrame, current_user_key, user_type)

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

    def forget_page(self):
        for frame in self.mainFrame.winfo_children():
            frame.grid_forget()

    def call_page(self, btn, page):
        self.sidebar.reset_indicator()
        if btn is not None:
            # For CustomTkinter, we use different methods to indicate selection
            btn.configure(border_width=2, border_color="#F1F1F1")

        # Identify current active page and clean it up if needed
        current_active = None
        for frame in self.mainFrame.winfo_children():
            if frame.winfo_viewable():
                current_active = frame
                break

        # Clean up if it's the report page
        if current_active is self.report and hasattr(self.report, 'cleanup'):
            print("Cleaning up Water Quality Report before switching pages")
            self.report.cleanup()

        # Now forget and show the new page
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


if __name__ == "__main__":
    app = Main(current_user_key)
    app.mainloop()
