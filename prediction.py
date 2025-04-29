import customtkinter as ctk


class PredictionPage(ctk.CTkFrame):
    def __init__(self, parent, bg_color=None):
        super().__init__(parent, fg_color=bg_color or "transparent")
        self.parent = parent
        self.create_widgets()

    def create_widgets(self):
        predictionlb = ctk.CTkLabel(self, text="PREDICTION TOOL", font=("Arial", 25, "bold"))
        predictionlb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")