import tkinter as tk

class settingsPage(tk.Frame):
    def __init__(self, parent, bg="#F1F1F1"):
        super().__init__(parent, bg=bg)
        self.parent = parent
        self.create_widgets()
    
    def create_widgets(self):
        settingslb = tk.Label(self, text="SETTINGS", font=("Arial", 25, "bold"))
        settingslb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")


