from basePage import basePage
import tkinter as tk

class inputDataPage(basePage):
    def create_widgets(self):
        tk.Label(self, text="INPUT DATA", font=("Arial", 16)).pack(pady=20)