from basePage import basePage
import tkinter as tk

class predictionPage(basePage):
    def create_widgets(self):
        tk.Label(self, text="PREDICTION", font=("Arial", 16)).pack(pady=20)