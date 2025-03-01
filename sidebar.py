import tkinter as tk
from tkinter import PhotoImage

from dashboard import dashboardPage
from about import aboutPage
from inputData import inputDataPage
from waterQualRep import waterQualRep
from prediction import predictionPage
from settings import settingsPage

min_w = 85
cur_width = min_w
expanded = False
is_hovering = False

class sidebar(tk.Frame):
    def __init__(self, parent, controller, icon_manager, mainFrame):
        super().__init__(parent, bg = "#1d97bd", width = 85, height=parent.winfo_height())
        self.parent = parent
        self.controller = controller
        self.mainFrame = mainFrame
        self.propagate(False)

        self.icon_manager = icon_manager
        self.AboutIcon = self.icon_manager.get_icon("AboutIcon")
        self.AppIcon = self.icon_manager.get_icon("AppIcon")
        self.AppLogo = self.icon_manager.get_icon("AppLogo")
        self.DBIcon = self.icon_manager.get_icon("DBIcon")
        self.IPIcon = self.icon_manager.get_icon("IPIcon")
        self.WQRIcon = self.icon_manager.get_icon("WQRIcon")
        self.PTIcon = self.icon_manager.get_icon("PTIcon")
        self.SIcon = self.icon_manager.get_icon("SIcon")


        self.min_w = 85
        self.max_w = 200
        self.cur_width = self.min_w
        self.expanded = False
        self.is_hovering = False

        self.grid(row = 0, column=0, sticky = "ns")

        self.btn1 = tk.Button(self, image=self.AppIcon, cursor="hand2", anchor="center", bg="#F1F1F1", relief="flat" , fg="#F1F1F1", command=lambda: self.controller.call_page(self.btn1, self.controller.about.show))
        self.btn1.image = self.AppIcon
        
        self.btn2 = tk.Button(self, image=self.DBIcon, cursor="hand2", anchor="center", bg="#F1F1F1", relief="flat" , fg="#F1F1F1", command=lambda: self.controller.call_page(self.btn2, self.controller.dashboard.show))
        self.btn2.image = self.DBIcon
        
        self.btn3 = tk.Button(self, image=self.IPIcon, cursor="hand2", anchor="center", bg="#F1F1F1", relief="flat" , fg="#F1F1F1", command=lambda: self.controller.call_page(self.btn3, self.controller.input.show))
        self.btn3.image = self.IPIcon
        
        self.btn4 = tk.Button(self, image=self.WQRIcon, cursor="hand2", anchor="center", bg="#F1F1F1", relief="flat" , fg="#F1F1F1", command=lambda: self.controller.call_page(self.btn4, self.controller.report.show))
        self.btn4.image = self.WQRIcon
        
        self.btn5 = tk.Button(self, image=self.PTIcon, cursor="hand2", anchor="center", bg="#F1F1F1", relief="flat" , fg="#F1F1F1", command=lambda: self.controller.call_page(self.btn5, self.controller.predict.show))
        self.btn5.image = self.PTIcon
        
        self.btn6 = tk.Button(self, image=self.SIcon, cursor="hand2", anchor="center", bg="#F1F1F1", relief="flat" , fg="#F1F1F1", command=lambda: self.controller.call_page(self.btn6, self.controller.settings.show))
        self.btn6.image = self.SIcon

        self.btn1.pack(fill="x", padx=20, pady=20)
        self.btn2.pack(fill="x", padx=20, pady=10)
        self.btn3.pack(fill="x", padx=20, pady=10)
        self.btn4.pack(fill="x", padx=20, pady=10)
        self.btn5.pack(fill="x", padx=20, pady=10)
        self.btn6.pack(side="bottom", fill="x", padx=20, pady=25)

        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.parent.bind("<Configure>", self.update_size)

        for self.btn in [self.btn1, self.btn2, self.btn3, self.btn4, self.btn5, self.btn6]:
            self.btn.bind('<Enter>', self.on_enter)
            self.btn.bind('<Leave>', self.on_leave)

    def reset_indicator(self):
        self.btn1.config(relief="flat")
        self.btn2.config(relief="flat")
        self.btn3.config(relief="flat")
        self.btn4.config(relief="flat")
        self.btn5.config(relief="flat")
        self.btn6.config(relief="flat")
    
    def expand(self):
        global cur_width, expanded
        self.cur_width += 5
        self.rep = self.parent.after(5, self.expand)
        self.config(width=self.cur_width)

        if self.cur_width >= self.max_w:
            self.expanded = True
            self.parent.after_cancel(self.rep)
            self.cur_width = self.max_w
            self.fill()
    
    def contract(self):
        global cur_width, expanded
        self.cur_width -= 5
        self.rep = self.parent.after(5, self.contract)
        self.config(width=self.cur_width)

        if self.cur_width <= self.min_w:
            self.expanded = False
            self.parent.after_cancel(self.rep)
            self.cur_width = self.min_w
            self.fill()

    def fill(self):
        global expanded
        if self.expanded:
            self.btn1.config(text="", image=self.AppIcon, width= 20, bg = "#FFFFFF")
            self.btn2.config(text=" DASHBOARD", image=self.DBIcon, compound="left", fg="#f1f1f1", width=20, bg="#1d97bd", anchor="w", padx=10)
            self.btn3.config(text=" INPUT DATA", image=self.IPIcon, compound="left", fg="#f1f1f1", width=20, bg="#1d97bd", anchor="w", padx=10)
            self.btn4.config(text=" WATER QUALITY\nREPORT", image=self.WQRIcon, compound="left", fg="#f1f1f1", width=20, bg="#1d97bd", anchor="w", padx=10)
            self.btn5.config(text=" PREDICTION TOOL", image=self.PTIcon, compound="left", fg="#f1f1f1", width=20, bg="#1d97bd", anchor="w", padx=10)
            self.btn6.config(text=" SETTINGS", image=self.SIcon, compound="left", fg="#f1f1f1", width=20, bg="#1d97bd", anchor="w", padx=10)
        else:
            self.btn1.config(text="L", width=5)
            self.btn2.config(text="", image=self.DBIcon, width=10, anchor="center")
            self.btn3.config(text="", image=self.IPIcon, width=10, anchor="center")
            self.btn4.config(text="", image=self.WQRIcon, width=10, anchor="center")
            self.btn5.config(text="", image=self.PTIcon, width=10, anchor="center")
            self.btn6.config(text="", image=self.SIcon, width=10, anchor="center")
    
    def on_enter(self, event):
        global is_hovering
        self.is_hovering = True
        if not self.expanded:
            self.expand()
    
    def on_leave(self, event):
        global is_hovering
        self.is_hovering = False
        self.parent.after(100, self.check_and_contract)

    def check_and_contract(self):
        if not self.is_hovering:
            self.contract()
    
    def update_size(self, event = None):
        self.config(height= self.parent.winfo_height())
        self.mainFrame.config(height =self.parent.winfo_height(), width = (self.parent.winfo_width()))