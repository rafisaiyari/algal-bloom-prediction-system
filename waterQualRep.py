from basePage import basePage
import tkinter as tk

class waterQualRepPage(basePage):
    def create_widgets(self):
        tk.Label(self, text="WATER QUALITY REPORT", font=("Arial", 16)).pack(pady=20)