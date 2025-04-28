import customtkinter as ctk

min_w = 85
cur_width = min_w
expanded = False
is_hovering = False


class Sidebar(ctk.CTkFrame):
    def __init__(self, parent, controller, icon_manager, mainFrame, user_type="regular"):
        super().__init__(parent, fg_color="#1d97bd", width=85, height=parent.winfo_height())
        self.user_type = user_type
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

        self.grid(row=0, column=0, sticky="ns")

        # Changed all Button widgets to CTkButton
        self.btn1 = ctk.CTkButton(self, image=self.AppIcon, text="", cursor="hand2",
                                  anchor="center", fg_color="#F1F1F1", hover_color="#E0E0E0",
                                  command=lambda: self.controller.call_page(self.btn1, self.controller.about.show))

        self.btn2 = ctk.CTkButton(self, image=self.DBIcon, text="", cursor="hand2",
                                  anchor="center", fg_color="#1d97bd", hover_color="#1a86a8",
                                  command=lambda: self.controller.call_page(self.btn2, self.controller.dashboard.show))

        self.btn3 = ctk.CTkButton(self, image=self.IPIcon, text="", cursor="hand2",
                                  anchor="center", fg_color="#1d97bd", hover_color="#1a86a8",
                                  command=lambda: self.controller.call_page(self.btn3, self.controller.input.show))

        if self.user_type == "regular":
            self.btn3.configure(state="disabled")

        self.btn4 = ctk.CTkButton(self, image=self.WQRIcon, text="", cursor="hand2",
                                  anchor="center", fg_color="#1d97bd", hover_color="#1a86a8",
                                  command=lambda: self.controller.call_page(self.btn4, self.controller.report.show))

        self.btn5 = ctk.CTkButton(self, image=self.PTIcon, text="", cursor="hand2",
                                  anchor="center", fg_color="#1d97bd", hover_color="#1a86a8",
                                  command=lambda: self.controller.call_page(self.btn5, self.controller.predict.show))

        self.btn6 = ctk.CTkButton(self, image=self.SIcon, text="", cursor="hand2",
                                  anchor="center", fg_color="#1d97bd", hover_color="#1a86a8",
                                  command=lambda: self.controller.call_page(self.btn6, self.controller.settings.show))

        self.btn1.pack(fill="x", padx=20, pady=20)
        self.btn2.pack(fill="x", padx=20, pady=15)
        self.btn3.pack(fill="x", padx=20, pady=15)
        self.btn4.pack(fill="x", padx=20, pady=15)
        self.btn5.pack(fill="x", padx=20, pady=15)
        self.btn6.pack(side="bottom", fill="x", padx=20, pady=25)

        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

        for self.btn in [self.btn1, self.btn2, self.btn3, self.btn4, self.btn5, self.btn6]:
            self.btn.bind('<Enter>', self.on_enter)
            self.btn.bind('<Leave>', self.on_leave)

    def reset_indicator(self):
        # In CustomTkinter, instead of manipulating relief, we use a more modern approach
        # by changing the button's fg_color (background)
        self.btn1.configure(fg_color="#F1F1F1", border_width=0)
        self.btn2.configure(fg_color="#1d97bd", border_width=0)
        self.btn3.configure(fg_color="#1d97bd", border_width=0)
        self.btn4.configure(fg_color="#1d97bd", border_width=0)
        self.btn5.configure(fg_color="#1d97bd", border_width=0)
        self.btn6.configure(fg_color="#1d97bd", border_width=0)

    def expand(self):
        self.cur_width += 5
        self.rep = self.parent.after(5, self.expand)
        self.configure(width=self.cur_width)

        if self.cur_width >= self.max_w:
            self.expanded = True
            self.parent.after_cancel(self.rep)
            self.cur_width = self.max_w
            self.fill()

    def contract(self):
        self.cur_width -= 5
        self.rep = self.parent.after(5, self.contract)
        self.configure(width=self.cur_width)

        if self.cur_width <= self.min_w:
            self.expanded = False
            self.parent.after_cancel(self.rep)
            self.cur_width = self.min_w
            self.fill()

    def fill(self):
        if self.expanded:
            self.btn1.configure(text="", image=self.AppLogo, width=20, height=35, fg_color="#FFFFFF")
            self.btn2.configure(text=" DASHBOARD", image=self.DBIcon, compound="left", text_color="#f1f1f1", width=20,
                                fg_color="#1d97bd", anchor="w")
            self.btn3.configure(text=" INPUT DATA", image=self.IPIcon, compound="left", text_color="#f1f1f1", width=20,
                                fg_color="#1d97bd", anchor="w")
            self.btn4.configure(text=" WATER QUALITY\nREPORT", image=self.WQRIcon, compound="left",
                                text_color="#f1f1f1",
                                width=20, fg_color="#1d97bd", anchor="w")
            self.btn5.configure(text=" PREDICTION TOOL", image=self.PTIcon, compound="left", text_color="#f1f1f1",
                                width=20, fg_color="#1d97bd", anchor="w")
            self.btn6.configure(text=" SETTINGS", image=self.SIcon, compound="left", text_color="#f1f1f1",
                                width=20, fg_color="#1d97bd", anchor="w")
        else:
            self.btn1.configure(text="", image=self.AppIcon, width=35, height=35)
            self.btn2.configure(text="", image=self.DBIcon, width=35, anchor="center")
            self.btn3.configure(text="", image=self.IPIcon, width=35, anchor="center")
            self.btn4.configure(text="", image=self.WQRIcon, width=35, anchor="center")
            self.btn5.configure(text="", image=self.PTIcon, width=35, anchor="center")
            self.btn6.configure(text="", image=self.SIcon, width=35, anchor="center")

    def on_enter(self, event):
        self.is_hovering = True
        if not self.expanded:
            self.expand()

    def on_leave(self, event):
        self.is_hovering = False
        self.parent.after(100, self.check_and_contract)

    def check_and_contract(self):
        if not self.is_hovering:
            self.contract()