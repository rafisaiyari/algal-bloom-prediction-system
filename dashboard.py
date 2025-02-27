from basePage import basePage
import tkinter as tk

class dashboardPage(basePage):
    def create_widgets(self):
        tk.Label(self, text="DASHBOARD", font=("Arial", 16)).pack(pady=20)
