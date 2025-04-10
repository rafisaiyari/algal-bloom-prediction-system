import tkinter as tk

from sidebar import Sidebar
from dashboard import DashboardPage
from about import AboutPage
from inputData import InputDataPage
from waterQualityReport import WaterQualityReport
from prediction import PredictionPage
from settings import SettingsPage
from icons import IconManager
from globals import current_user_key


class Main(tk.Tk):
    def __init__(self, current_user_key, user_type="regular"):
        super().__init__()
        self.current_user_key = current_user_key
        self.user_type = user_type
        print(f"User: {current_user_key}, Type: {user_type}")
        self.title("Bloom Sentry")

        self.minsize(800, 600)
        self.geometry('1280x720')
        self.propagate(False)
        self.rowconfigure(0, weight=1)

        # Mainframe
        self.mainFrame = tk.Frame(self, width=(self.winfo_width() - 100), height=self.winfo_height(), bg="#F1F1F1")
        self.mainFrame.grid(row=0, column=1, sticky="nsew")
        self.mainFrame.rowconfigure(0, weight=1)
        self.mainFrame.columnconfigure(0, weight=1)
        # Scaling from login.py to main.py
        self.mainFrame.tk.call('tk', 'scaling', 1.2)

        # Instantiate Sidebar, and Icon Manager
        self.icon_manager = IconManager()
        self.sidebar = Sidebar(self, self, self.icon_manager, self.mainFrame)

        self.dashboard = DashboardPage(self.mainFrame)
        self.about = AboutPage(self.mainFrame)
        self.input = InputDataPage(self.mainFrame)
        self.report = WaterQualityReport(self.mainFrame)
        self.predict = PredictionPage(self.mainFrame)
        self.settings = SettingsPage(self.mainFrame, current_user_key, user_type)

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
    app = Main(current_user_key)
    app.mainloop()