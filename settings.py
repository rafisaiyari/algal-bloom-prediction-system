from basePage import basePage
import tkinter as tk

class settingsPage(basePage):
    def create_widgets(self):
        tk.Label(self, text="SETTINGS", font=("Arial", 16)).pack(pady=20)