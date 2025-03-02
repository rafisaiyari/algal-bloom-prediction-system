import tkinter as tk

from sidebar import Sidebar
from dashboard import DashboardPage
from about import AboutPage
from inputData import InputDataPage
from waterQualRep import WaterQualRep
from prediction import PredictionPage
from settings import SettingsPage
from icons import IconManager
from login import LoginApp

class Main(tk.Tk):
    def __init__(self, current_user_key=None):
        super().__init__()
        self.title("Bloom Sentry")

        self.minsize(800, 600)
        self.geometry('1280x720')
        self.propagate(False)

        # Mainframe
        self.mainFrame = tk.Frame(self, width=(self.winfo_width() - 100), height=self.winfo_height(), bg="#F1F1F1")
        self.mainFrame.grid(row=0, column=1, sticky="nsew")
        self.mainFrame.rowconfigure(0, weight=1)
        self.mainFrame.columnconfigure(0, weight=1)

        # Instantiate Sidebar, and Icon Manager
        self.icon_manager = IconManager()
        self.sidebar = Sidebar(self, self, self.icon_manager, self.mainFrame)
        self.dashboard = DashboardPage(self.mainFrame)
        self.about = AboutPage(self.mainFrame)
        self.input = InputDataPage(self.mainFrame)
        self.report = WaterQualRep(self.mainFrame)
        self.predict = PredictionPage(self.mainFrame)
        self.settings = SettingsPage(self.mainFrame, current_user_key)
        self.login = LoginApp
        self.login_window()

    def login_window(self):
        """Open the login window."""
        login_app = LoginApp()  # Create an instance of LoginApp
        self.wait_window(login_app.app)  # Wait for the login window to close

        # After login, you can check if the user is logged in and proceed
        if login_app.current_user_key:  # Assuming current_user_key is set in LoginApp
            self.call_page(None, self.dashboard.show)

    def forget_page(self):
        for frame in self.mainFrame.winfo_children():
            frame.grid_forget()

    def call_page(self, btn, page):
        self.sidebar.reset_indicator()
        if btn is not None:
            btn.config(relief="ridge", highlightbackground="#F1F1F1", highlightthickness=2)
        self.forget_page()
        page()


if __name__ == "__main__":
    app = Main()
    app.mainloop()

