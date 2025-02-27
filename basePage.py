import tkinter as tk

class basePage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#FFFFFF")
        self.parent = parent
        self.create_widgets()

    def create_widgets(self):
        pass