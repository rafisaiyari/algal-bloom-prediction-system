import customtkinter as ctk
from PIL import Image


class IconManager:
    def __init__(self):
        # Dictionary to store icons
        self.icons = {}

        # Load all icons
        self.load_icon("AboutIcon", "Icons/AppLogo.png", (200, 65))
        self.load_icon("AppIcon", "Icons/AppIcon.png", (35, 35))
        self.load_icon("AppLogo", "Icons/AppLogo.png", (120, 35))

        self.load_icon("DBIcon", "Icons/DBIcon.png", (25, 25))
        self.load_icon("IPIcon", "Icons/IPIcon.png", (25, 25))
        self.load_icon("WQRIcon", "Icons/WQRIcon.png", (25, 25))
        self.load_icon("PTIcon", "Icons/PTIcon.png", (25, 25))
        self.load_icon("SIcon", "Icons/SIcon.png", (25, 25))

    def load_icon(self, name, path, size):
        try:
            # Create CTkImage instead of ImageTk.PhotoImage
            # This properly handles high DPI displays
            self.icons[name] = ctk.CTkImage(
                light_image=Image.open(path),
                dark_image=Image.open(path),
                size=size
            )
        except Exception as e:
            print(f"Error loading icon '{name}' from '{path}': {e}")

    def get_icon(self, icon_name):
        return self.icons[icon_name]