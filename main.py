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

from sidebar import sidebarFrame
from dashboard import dashboardPage
from about import aboutPage
from inputData import inputDataPage
from waterQualRep import waterQualRepPage
from prediction import predictionPage
from settings import settingsPage

class main(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bloom Sentry")
        self.geometry("800x600")
        
        self.pages = {}
        self.current_page = None
        self.show_frame = self.show_frame
        self.sidebar = sidebarFrame(self, self.show_frame)
        
        
        self.grid_columnconfigure(1, weight=1)  # Allow mainFrame to expand
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = sidebarFrame(self, self.show_frame)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.mainFrame = tk.Frame(self, bg = "#F1F1F1")
        self.mainFrame.grid(row=0, column=1, sticky="nsew")

        self.pages = {
            "dashboardPage" : dashboardPage(self.mainFrame),
            "aboutPage" : aboutPage(self.mainFrame),
            "inputDataPage" : inputDataPage(self.mainFrame),
            "waterQualRepPage" : waterQualRepPage(self.mainFrame),
            "predictionPage" : predictionPage(self.mainFrame),
            "settingsPage" : settingsPage(self.mainFrame),
        }

        self.show_frame("dashboardPage")
    
    def show_frame(self, page_name):
        if self.current_page:
            self.current_page.forget()
        
        self.current_page = self.pages[page_name]
        self.current_page.grid(row=0, column=0, sticky="nsew")

if __name__ == "__main__":
    app = main()
    app.mainloop()

