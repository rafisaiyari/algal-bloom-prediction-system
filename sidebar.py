import tkinter as tk
from tkinter import PhotoImage

from dashboard import dashboardPage
from about import aboutPage
from inputData import inputDataPage
from waterQualRep import waterQualRepPage
from prediction import predictionPage
from settings import settingsPage

class sidebarFrame(tk.Frame):
    def __init__(self, parent, callpage):
        super().__init__(parent, bg = "#1d97bd", width = 85, height=parent.winfo_height())
        self.parent = parent
        self.callpage = callpage
        
        self.min_w = 90
        self.max_w = 200
        self.cur_width = self.min_w
        self.expanded = False
        self.is_hovering = False
        self.animJob = None

        self.navBar = tk.Frame(self, bg="#1d97bd", width=self.cur_width, height=parent.winfo_height())
        self.navBar.grid(row=0, column=0, sticky="ns")
        self.navBar.propagate(False)
        
        self.logoIcon = PhotoImage(file = "Icons/SIcon.png")
        self.DBIcon = PhotoImage(file = "Icons/DBIcon.png")
        self.IPIcon = PhotoImage(file = "Icons/IPIcon.png")
        self.WQRIcon = PhotoImage(file = "Icons/WQRIcon.png")
        self.PTIcon = PhotoImage(file = "Icons/PTIcon.png")
        self.SIcon = PhotoImage(file = "Icons/SIcon.png")

        self.btn1 = tk.Button(self.navBar, text = "L", width= 5, height=2, relief="flat", bg="#1d97bd",fg="white", command=lambda: self.callpage(None, aboutPage))
        self.btn2 = tk.Button(self.navBar, image = self.DBIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: self.callpage(None, dashboardPage))
        self.btn3 = tk.Button(self.navBar, image = self.IPIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: self.callpage(None, inputDataPage))
        self.btn4 = tk.Button(self.navBar, image = self.WQRIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: self.callpage(None, waterQualRepPage))
        self.btn5 = tk.Button(self.navBar, image = self.PTIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: self.callpage(None, predictionPage))
        self.btn6 = tk.Button(self.navBar, image = self.SIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: self.callpage(None, settingsPage))

        self.btn1.pack(fill="x", padx=20, pady=20)
        self.btn2.pack(fill="x", padx=20, pady=10)
        self.btn3.pack(fill="x", padx=20, pady=10)
        self.btn4.pack(fill="x", padx=20, pady=10)
        self.btn5.pack(fill="x", padx=20, pady=10)
        self.btn6.pack(side="bottom", fill="x", padx=20, pady=25)

        for btn in [self.btn1, self.btn2, self.btn3, self.btn4, self.btn5, self.btn6]:
            self.bind("<Enter>", self.on_enter)
            self.bind("<Leave>", self.on_leave)
        
        self.bind("<Configure>", self.update_sidebar_height)
        
        self.mainFrame = tk.Frame(self.parent, bg="#f1f1f1")
        self.mainFrame.grid(row=0, column=1, sticky="nsew")
    
        
    def expand(self):
        self.cur_width += 10
        self.animJob = self.after(5, self.expand)
        self.navBar.config(width=self.cur_width)
        
        if self.cur_width >= self.max_w:
            self.expanded = True
            self.after_cancel(self.animJob)
            self.cur_width = self.max_w
            self.fill()

    def contract(self):
        self.cur_width -= 10
        self.animJob = self.after(5, self.contract)
        self.navBar.config(width=self.cur_width)
        
        if self.cur_width <= self.min_w:
            self.expanded = False
            self.after_cancel(self.animJob)
            self.cur_width = self.min_w
            self.after(5, self.contract)
            self.fill()


    def fill(self):
        if self.expanded:
            self.btn1.config(text=" LOGO", width= 20)
            self.btn2.config(text=" DASHBOARD", image=self.DBIcon, compound="left", fg="#f1f1f1", width=20, bg="#1d97bd", anchor="w", padx=10)
            self.btn3.config(text=" INPUT DATA", image=self.IPIcon, compound="left", fg="#f1f1f1", width=20, bg="#1d97bd", anchor="w", padx=10)
            self.btn4.config(text=" WATER QUALITY\nREPORT", image=self.WQRIcon, compound="left", fg="#f1f1f1", width=20, bg="#1d97bd", anchor="w", padx=10)
            self.btn5.config(text=" PREDICTION TOOL", image=self.PTIcon, compound="left", fg="#f1f1f1", width=20, bg="#1d97bd", anchor="w", padx=10)
            self.btn6.config(text=" SETTINGS", image=self.SIcon, compound="left", fg="#f1f1f1", width=20, bg="#1d97bd", anchor="w", padx=10)
        else:
            self.btn1.config(text="L", width=5)
            self.btn2.config(text="", image=self.DBIcon, width=10)
            self.btn3.config(text="", image=self.IPIcon, width=10)
            self.btn4.config(text="", image=self.WQRIcon, width=10)
            self.btn5.config(text="", image=self.PTIcon, width=10)
            self.btn6.config(text="", image=self.SIcon, width=10)

    def on_enter(self, event):
        self.is_hovering = True
        if not self.expanded:
            self.expand()

    def on_leave(self, event):
        self.is_hovering = False
        self.after(100, self.check_and_contract)
    
    def check_and_contract(self):
        if not self.is_hovering:
            self.contract()
    
    def update_sidebar_height(self, event = None):
        self.navBar.config(height=self.parent.winfo_height())

    def reset_indicator(self):
        self.btn1.config(relief="flat")
        self.btn2.config(relief="flat")
        self.btn3.config(relief="flat")
        self.btn4.config(relief="flat")
        self.btn5.config(relief="flat")
        self.btn6.config(relief="flat")


