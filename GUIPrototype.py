import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.geometry('1280x720')

min_w = 80   # Minimum width of the navBar
max_w = 200  # Maximum width of the navBar
cur_width = min_w  # Initial width
expanded = False  # Expansion flag

is_hovering = False  # Track mouse hover state

# Function to expand navbar
def expand():
    global cur_width, expanded
    cur_width += 10
    rep = root.after(5, expand)
    navBar.config(width=cur_width)
    
    if cur_width >= max_w:
        expanded = True
        root.after_cancel(rep)
        cur_width = max_w
        fill()

# Function to contract navbar
def contract():
    global cur_width, expanded
    cur_width -= 10
    rep = root.after(5, contract)
    navBar.config(width=cur_width)

    if cur_width <= min_w:
        expanded = False
        root.after_cancel(rep)
        cur_width = min_w
        fill()

# Function to update button layout
def fill():
    global expanded
    if expanded:
        bg1.config(text=" LOGO", image="", width=20, bg="#1d97bd")
        bg2.config(text=" DASHBOARD", image=DBIcon, compound="left", fg="#F1F1F1", width=20, bg="#1d97bd", anchor="w", padx=10)
        bg3.config(text=" INPUT DATA", image=IPIcon, compound="left", fg="#F1F1F1", width=20, bg="#1d97bd", anchor="w", padx=10)
        bg4.config(text=" WATER QUALITY\nREPORT", image=WQRIcon, compound="left", fg="#F1F1F1", width=20, height=30, bg="#1d97bd", anchor="w", padx=10)
        bg5.config(text=" PREDICTION TOOL", image=PTIcon, compound="left", fg="#F1F1F1", width=20, bg="#1d97bd", anchor="w", padx=10)
        bg6.config(text=" SETTINGS", image=SIcon, compound="left", fg="#F1F1F1", width=20, bg="#1d97bd", anchor="w", padx=10)
    else:
        bg1.config(text="L", width=5)
        bg2.config(text="", image=DBIcon, width=5)
        bg3.config(text="", image=IPIcon, width=5)
        bg4.config(text="", image=WQRIcon, width=5)
        bg5.config(text="", image=PTIcon, width=5)
        bg6.config(text="", image=SIcon, width=5)

# Expand when mouse enters
def on_enter(e):
    global is_hovering
    is_hovering = True
    if not expanded:
        expand()

# Contract when mouse leaves
def on_leave(e):
    global is_hovering
    is_hovering = False
    root.after(100, check_and_contract)

def check_and_contract():
    if not is_hovering:
        contract()

def update_sidebar_height(event=None):
    navBar.config(height=root.winfo_height())

# Bind resize event
root.bind("<Configure>", update_sidebar_height)

# Sidebar Frame
navBar = tk.Frame(root, width=cur_width, height=root.winfo_height(), bg="#1d97bd")
navBar.grid(row=0, column=0, sticky="ns")

# Load icons
DBIcon = ImageTk.PhotoImage(Image.open('Icons/DBIcon.png').resize((25,25)))
IPIcon = ImageTk.PhotoImage(Image.open('Icons/IPIcon.png').resize((25,25)))
WQRIcon = ImageTk.PhotoImage(Image.open('Icons/WQRIcon.png').resize((25,25)))
PTIcon = ImageTk.PhotoImage(Image.open('Icons/PTIcon.png').resize((25,25)))
SIcon = ImageTk.PhotoImage(Image.open('Icons/SIcon.png').resize((25,25)))

# Define buttons
bg1 = tk.Button(navBar, text="L", width=5, height=2, relief="flat", bg="#1d97bd", fg="#F1F1F1")
bg2 = tk.Button(navBar, image=DBIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1")
bg3 = tk.Button(navBar, image=IPIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1")
bg4 = tk.Button(navBar, image=WQRIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1")
bg5 = tk.Button(navBar, image=PTIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1")
bg6 = tk.Button(navBar, image=SIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1")

# Pack buttons with equal width
bg1.pack(fill="x", padx=20, pady=20)
bg2.pack(fill="x", padx=20, pady=10)
bg3.pack(fill="x", padx=20, pady=10)
bg4.pack(fill="x", padx=20, pady=10)
bg5.pack(fill="x", padx=20, pady=10)
bg6.pack(side="bottom", fill="x", padx=20, pady=25)

# Bind hover events
navBar.bind('<Enter>', on_enter)
navBar.bind('<Leave>', on_leave)

for btn in [bg1, bg2, bg3, bg4, bg5, bg6]:
    btn.bind('<Enter>', on_enter)
    btn.bind('<Leave>', on_leave)

navBar.propagate(False)
root.mainloop()
