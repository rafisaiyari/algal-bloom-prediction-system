import tkinter as tk
import pandas as pd
import csv
import os

from tkinter import ttk
from tkinter import Label, Tk, Entry
from tkcalendar import DateEntry
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sidebar import sidebar
from dashboard import dashboardPage
from about import aboutPage
from inputData import inputDataPage
from waterQualRep import reportPage
from prediction import predictionPage
from settings import settingsPage
from icons import iconManager

class main(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bloom Sentry")
        
        self.minsize(800, 600)
        self.geometry('1280x720')


        # Mainframe
        self.mainFrame = tk.Frame(self, width=(self.winfo_width() - 100), height=self.winfo_height(), bg="#F1F1F1")
        self.mainFrame.grid(row=0, column=1, sticky="nsew")

        #Instantiate Sidebar, and Icon Manager
        self.icon_manager = iconManager()
        self.sidebar = sidebar(self, self, self.icon_manager, self.mainFrame)

        self.dashboard = dashboardPage(self.mainFrame)
        self.about = aboutPage(self.mainFrame)
        self.input = inputDataPage(self.mainFrame)
        self.report = reportPage(self.mainFrame)
        self.predict = predictionPage(self.mainFrame)
        self.settings = settingsPage(self.mainFrame)
        

    def forget_page(self):
        for frame in self.mainFrame.winfo_children():
            frame.grid_forget()

    def call_page(self, btn, page):
        self.sidebar.reset_indicator()
        if btn is not None:
            btn.config(relief="ridge", highlightbackground="#F1F1F1", highlightthickness=2)
        self.forget_page()
        if page is not None:
            page()

if __name__ == "__main__":
    app = main()
    app.mainloop()

