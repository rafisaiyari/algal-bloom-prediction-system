from basePage import basePage
import tkinter as tk

class aboutPage(basePage):
    def create_widgets(self):
        tk.Label(self, text="ABOUT US", font=("Arial", 16)).pack(pady=20)