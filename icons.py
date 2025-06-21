import customtkinter as ctk
from PIL import Image
import os
import sys


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

    def get_resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def load_icon(self, name, path, size):
        try:
            full_path = self.get_resource_path(path)

            self.icons[name] = ctk.CTkImage(
                light_image=Image.open(full_path),
                dark_image=Image.open(full_path),
                size=size
            )
        except Exception as e:
            print(f"Error loading icon '{name}' from '{path}': {e}")
            if self.icons and "AppLogo" in self.icons:
                self.icons[name] = self.icons["AppLogo"]
            elif self.icons:
                self.icons[name] = next(iter(self.icons.values()))

    def get_icon(self, icon_name):
        if icon_name in self.icons:
            return self.icons[icon_name]
        else:
            print(f"Warning: Icon '{icon_name}' not found")
            return self.icons.get("AppLogo") if "AppLogo" in self.icons else None