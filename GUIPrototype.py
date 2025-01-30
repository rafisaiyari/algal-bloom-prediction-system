import tkinter as tk
from PIL import Image, ImageTk  # Import Pillow for resizing images

root = tk.Tk()
root.geometry('1280x720')

min_w = 70  # Minimum width of the navBar
max_w = 200  # Maximum width of the navBar
max_h = 1920
cur_width = min_w  # Current width of the navBar
expanded = False  # Check if it is completely expanded

canvas_widgets = []

is_hovering = False  # Flag to track if the mouse is inside the sidebar


# Resize images to fit buttons (33 x 30 pixels)
def resize_image(path, width, height):
    img = Image.open(path)
    img = img.resize((int(width), int(height)), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


# Load and resize images
img_logo = resize_image('C:/Users/Legion 5 Pro/Desktop/algal_bloom_prediction/buttons/dashboard_button.png', 33, 30)
img_dashboard = resize_image('C:/Users/Legion 5 Pro/Desktop/algal_bloom_prediction/buttons/dashboard_button.png', 33, 30)
img_input_data = resize_image('C:/Users/Legion 5 Pro/Desktop/algal_bloom_prediction/buttons/dashboard_button.png', 33, 30)
img_water_quality = resize_image('C:/Users/Legion 5 Pro/Desktop/algal_bloom_prediction/buttons/dashboard_button.png', 33, 30)
img_prediction_tool = resize_image('C:/Users/Legion 5 Pro/Desktop/algal_bloom_prediction/buttons/dashboard_button.png', 33, 30)


def expand():
    global cur_width, expanded
    cur_width += 1  # Increase the width by 10
    rep = root.after(1, expand)  # Repeat this func every 5 ms
    navBar.config(width=cur_width)  # Change the width to new increased width
    if cur_width >= max_w:  # If width is greater than maximum width
        expanded = True  # Frame is expanded
        root.after_cancel(rep)  # Stop repeating the func
        fill()


def contract():
    global cur_width, expanded
    cur_width -= 1  # Reduce the width by 10
    rep = root.after(1, contract)  # Call this func every 5 ms
    navBar.config(width=cur_width)  # Change the width to new reduced width
    if cur_width <= min_w:  # If it is back to normal width
        expanded = False  # Frame is not expanded
        root.after_cancel(rep)  # Stop repeating the func
        fill()





def fill():
    global expanded

    if expanded:
        bg1.config(text="LOGO", image="", compound="none")
        bg2.config(text="DASHBOARD", image="", compound="none")
        bg3.config(text="INPUT DATA", image="", compound="none")
        bg4.config(text="WATER QUALITY REPORT", justify="center", image="", compound="none")
        bg5.config(text="PREDICTION TOOL", image="", compound="none")

    else:
        bg1.config(image=img_logo, text="", compound="top")
        bg2.config(image=img_dashboard, text="", compound="top")
        bg3.config(image=img_input_data, text="", compound="top")
        bg4.config(image=img_water_quality, text="", compound="top")
        bg5.config(image=img_prediction_tool, text="", compound="top")


def on_enter(e):
    global is_hovering
    is_hovering = True
    if not expanded:
        expand()


def on_leave(e):
    global is_hovering
    is_hovering = False
    root.after(100, check_and_contract)


def check_and_contract():
    if not is_hovering:
        contract()


root.update()
navBar = tk.Frame(root, width=cur_width, height=max_h, bg="#1d97bd")
navBar.grid(row=0, column=0, sticky='ns')   # Allow vertical stretching

# Create buttons initially with images for contracted state
bg1 = tk.Button(navBar, image=img_logo, relief="flat")
bg2 = tk.Button(navBar, image=img_dashboard, relief="flat")
bg3 = tk.Button(navBar, image=img_input_data, relief="flat")
bg4 = tk.Button(navBar, image=img_water_quality, relief="flat")
bg5 = tk.Button(navBar, image=img_prediction_tool, relief="flat")

# Pack buttons with padding for spacing
bg1.pack(pady=20)
bg2.pack(pady=10)
bg3.pack(pady=10)
bg4.pack(pady=10)
bg5.pack(pady=10)

# Bind mouse enter and leave events to navBar and buttons
navBar.bind('<Enter>', on_enter)
navBar.bind('<Leave>', on_leave)

for button in [bg1, bg2, bg3, bg4, bg5]:
    button.bind('<Enter>', on_enter)
    button.bind('<Leave>', on_leave)

navBar.propagate(False)

root.mainloop()
