from PIL import Image, ImageTk

class iconManager:
    def __init__(self):
        
        # Dictionary to store icons
        self.icons = {}

        self.load_icon("AboutIcon", "Icons/AppLogo.png", (200,65))
        self.load_icon("AppIcon", "Icons/AppIcon.png", (35,35))
        self.load_icon("AppLogo", "Icons/AppLogo.png", (120,40))

        self.load_icon("DBIcon", "Icons/DBIcon.png", (25,25))
        self.load_icon("IPIcon", "Icons/IPIcon.png", (25,25))
        self.load_icon("WQRIcon", "Icons/WQRIcon.png", (25,25))
        self.load_icon("PTIcon", "Icons/PTIcon.png", (25,25))
        self.load_icon("SIcon", "Icons/SIcon.png", (25,25))

    def load_icon(self, name, path, size):
        self.icons[name] = ImageTk.PhotoImage(Image.open(path).resize(size))

    def get_icon(self, name):
        return self.icons.get(name, None)
    




