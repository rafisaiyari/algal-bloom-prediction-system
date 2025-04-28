import customtkinter as ctk
from PIL import Image


class Sidebar(ctk.CTkFrame):
    def __init__(self, parent, controller, icon_manager, mainFrame, user_type="regular"):
        super().__init__(parent, fg_color="#1d97bd", width=85, height=parent.winfo_height())
        self.user_type = user_type
        self.parent = parent
        self.controller = controller
        self.mainFrame = mainFrame
        self.propagate(False)

        # Initialize dimensions
        self.min_w = 85
        self.max_w = 200
        self.cur_width = self.min_w
        self.expanded = False
        self.is_hovering = False

        # Animation control
        self.animation_running = False
        self.animation_id = None

        # Load icons using icon_manager
        self.icon_manager = icon_manager
        self.AboutIcon = self.icon_manager.get_icon("AboutIcon")
        self.AppIcon = self.icon_manager.get_icon("AppIcon")
        self.AppLogo = self.icon_manager.get_icon("AppLogo")
        self.DBIcon = self.icon_manager.get_icon("DBIcon")
        self.IPIcon = self.icon_manager.get_icon("IPIcon")
        self.WQRIcon = self.icon_manager.get_icon("WQRIcon")
        self.PTIcon = self.icon_manager.get_icon("PTIcon")
        self.SIcon = self.icon_manager.get_icon("SIcon")

        self.grid(row=0, column=0, sticky="ns")

        # Create sidebar buttons
        self.create_buttons()

        # Set up hover events
        self.setup_bindings()

    def create_buttons(self):
        """Create all sidebar buttons"""
        # About button
        self.btn1 = ctk.CTkButton(
            self,
            image=self.AppIcon,
            text="",
            cursor="hand2",
            fg_color="#F1F1F1",
            hover_color="#E0E0E0",
            corner_radius=6,
            width=35,
            height=35,
            anchor="w",
            command=lambda: self.controller.call_page(self.btn1, self.controller.about.show)
        )

        # Dashboard button
        self.btn2 = ctk.CTkButton(
            self,
            image=self.DBIcon,
            text="",
            cursor="hand2",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            width=35,
            height=35,
            anchor="w",
            command=lambda: self.controller.call_page(self.btn2, self.controller.dashboard.show)
        )

        # Input Data button
        self.btn3 = ctk.CTkButton(
            self,
            image=self.IPIcon,
            text="",
            cursor="hand2",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            width=35,
            height=35,
            anchor="w",
            command=lambda: self.controller.call_page(self.btn3, self.controller.input.show)
        )

        # Disable button if user is "regular"
        if self.user_type == "regular":
            self.btn3.configure(state="disabled")

        # Water Quality Report button
        self.btn4 = ctk.CTkButton(
            self,
            image=self.WQRIcon,
            text="",
            cursor="hand2",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            width=35,
            height=35,
            anchor="w",
            command=lambda: self.controller.call_page(self.btn4, self.controller.report.show)
        )

        # Prediction Tool button
        self.btn5 = ctk.CTkButton(
            self,
            image=self.PTIcon,
            text="",
            cursor="hand2",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            width=35,
            height=35,
            anchor="w",
            command=lambda: self.controller.call_page(self.btn5, self.controller.predict.show)
        )

        # Settings button
        self.btn6 = ctk.CTkButton(
            self,
            image=self.SIcon,
            text="",
            cursor="hand2",
            fg_color="#1d97bd",
            hover_color="#1a86a8",
            corner_radius=6,
            width=35,
            height=35,
            anchor="w",
            command=lambda: self.controller.call_page(self.btn6, self.controller.settings.show)
        )

        # Place buttons
        self.btn1.pack(fill="x", padx=20, pady=20)
        self.btn2.pack(fill="x", padx=20, pady=15)
        self.btn3.pack(fill="x", padx=20, pady=15)
        self.btn4.pack(fill="x", padx=20, pady=15)
        self.btn5.pack(fill="x", padx=20, pady=15)
        self.btn6.pack(side="bottom", fill="x", padx=20, pady=25)

    def setup_bindings(self):
        """Set up mouse hover bindings"""
        # Bind mouse events to sidebar
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

        # Also bind to all buttons to ensure proper event capture
        for btn in [self.btn1, self.btn2, self.btn3, self.btn4, self.btn5, self.btn6]:
            btn.bind("<Enter>", self.on_enter)

    def reset_indicator(self):
        """Reset all button indicators to default state"""
        self.btn1.configure(fg_color="#F1F1F1", border_width=0)
        self.btn2.configure(fg_color="#1d97bd", border_width=0)
        self.btn3.configure(fg_color="#1d97bd", border_width=0)
        self.btn4.configure(fg_color="#1d97bd", border_width=0)
        self.btn5.configure(fg_color="#1d97bd", border_width=0)
        self.btn6.configure(fg_color="#1d97bd", border_width=0)

    def expand(self):
        """Expand the sidebar animation"""
        if self.animation_running:
            return

        self.animation_running = True

        def _expand_step():
            if self.cur_width < self.max_w:
                self.cur_width += 10
                self.configure(width=self.cur_width)
                self.animation_id = self.after(10, _expand_step)
            else:
                # Animation complete
                self.cur_width = self.max_w
                self.configure(width=self.max_w)
                self.expanded = True
                self.animation_running = False
                self.update_button_text(True)

        _expand_step()

    def contract(self):
        """Contract the sidebar animation"""
        if self.animation_running:
            return

        self.animation_running = True
        self.update_button_text(False)

        def _contract_step():
            if self.cur_width > self.min_w:
                self.cur_width -= 10
                self.configure(width=self.cur_width)
                self.animation_id = self.after(10, _contract_step)
            else:
                # Animation complete
                self.cur_width = self.min_w
                self.configure(width=self.min_w)
                self.expanded = False
                self.animation_running = False

        _contract_step()

    def update_button_text(self, show_text):
        """Update button text based on sidebar state"""
        if show_text:
            # Set text for expanded state while maintaining height
            self.btn1.configure(text="", image=self.AppLogo, width=160, height=35)  # Set height=35 to match AppIcon
            self.btn2.configure(text=" DASHBOARD", compound="left", text_color="#f1f1f1", width=160)
            self.btn3.configure(text=" INPUT DATA", compound="left", text_color="#f1f1f1", width=160)
            self.btn4.configure(text=" WATER QUALITY\nREPORT", compound="left", text_color="#f1f1f1", width=160)
            self.btn5.configure(text=" PREDICTION TOOL", compound="left", text_color="#f1f1f1", width=160)
            self.btn6.configure(text=" SETTINGS", compound="left", text_color="#f1f1f1", width=160)
        else:
            # Remove text for collapsed state and maintain height
            self.btn1.configure(text="", image=self.AppIcon, width=35, height=35)  # Keep height=35 consistent
            self.btn2.configure(text="", width=35)
            self.btn3.configure(text="", width=35)
            self.btn4.configure(text="", width=35)
            self.btn5.configure(text="", width=35)
            self.btn6.configure(text="", width=35)
    def on_enter(self, event):
        """Handle mouse enter events"""
        # Cancel any pending contract action
        if hasattr(self, "_pending_contract") and self._pending_contract:
            self.after_cancel(self._pending_contract)
            self._pending_contract = None

        self.is_hovering = True

        # Only expand if not already expanded
        if not self.expanded and not self.animation_running:
            self.expand()

    def on_leave(self, event):
        """Handle mouse leave events"""
        # Get the widget that the mouse is over
        widget_under_mouse = event.widget.winfo_containing(event.x_root, event.y_root)

        # Check if the mouse is still over any part of the sidebar
        if widget_under_mouse is not None:
            # Check if the widget is self or a child of self
            current = widget_under_mouse
            while current is not None:
                if current == self:
                    # Still inside sidebar, do nothing
                    return
                current = current.master

        # If we reach here, the mouse has truly left the sidebar
        self.is_hovering = False

        # Schedule contraction after a short delay
        if hasattr(self, "_pending_contract") and self._pending_contract:
            self.after_cancel(self._pending_contract)

        self._pending_contract = self.after(300, self.check_and_contract)

    def check_and_contract(self):
        """Check if mouse is still outside and contract sidebar if needed"""
        if not self.is_hovering and self.expanded and not self.animation_running:
            self.contract()
