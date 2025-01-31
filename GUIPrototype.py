import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.minsize(800, 600)
root.geometry('1280x720')

min_w = 85   # Minimum width of the navBar
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
        bg2.config(text="", image=DBIcon, width=10)
        bg3.config(text="", image=IPIcon, width=10)
        bg4.config(text="", image=WQRIcon, width=10)
        bg5.config(text="", image=PTIcon, width=10)
        bg6.config(text="", image=SIcon, width=10)

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

def reset_indicator():
    bg1.config(relief="flat")
    bg2.config(relief="flat")
    bg3.config(relief="flat")
    bg4.config(relief="flat")
    bg5.config(relief="flat")
    bg6.config(relief="flat")
    
def delete_page():
    for frame in mainFrame.winfo_children():
        frame.grid_forget()

def indicate(btn, page):
    reset_indicator()  # Call reset
    btn.config(relief="ridge", highlightbackground="#F1F1F1", highlightthickness=2)
    delete_page()
    page()  # Call the page function

def about_page():
    # Ensure the page appears with grid settings
    aboutpg.grid(row=0, column=1, sticky="nsew")

def dashboard_page():
    # Ensure the page appears with grid settings
    dashboardpg.grid(row=0, column=1, sticky="nsew")

def inputdata_page():
    # Ensure the page appears with grid settings
    inputdatapg.grid(row=0, column=1, sticky="nsew")

def waterreport_page():
    # Ensure the page appears with grid settings
    waterreportpg.grid(row=0, column=1, sticky="nsew")

def predictiontool_page():
    # Ensure the page appears with grid settings
    predictiontoolpg.grid(row=0, column=1, sticky="nsew")

def settings_page():
    # Ensure the page appears with grid settings
    settingspg.grid(row=0, column=1, sticky="nsew")

# Bind resize event
root.bind("<Configure>", update_sidebar_height)

# Sidebar Frame
navBar = tk.Frame(root, width=cur_width, height=root.winfo_height(), bg="#1d97bd")
navBar.grid(row=0, column=0, sticky="ns")

mainFrame = tk.Frame(root, bg="#F1F1F1")
mainFrame.grid(row=0, column=1, sticky="nsew")

# About Page Frame
aboutpg = tk.Frame(mainFrame, bg="#F1F1F1")
aboutlb = tk.Label(aboutpg, text="ABOUT US", font=("Comic Sans MS", 25, "bold"))
aboutlb.grid(row=0, column=0, padx=20, pady=20)
# Dashboard Page Frame
dashboardpg = tk.Frame(mainFrame, bg="#F1F1F1")
dashboardlb = tk.Label(dashboardpg, text="DASHBOARD", font=("Comic Sans MS", 25, "bold"))
dashboardlb.grid(row=0, column=0, padx=20, pady=20)
# About Page Frame
inputdatapg = tk.Frame(mainFrame, bg="#F1F1F1")
inputdatalb = tk.Label(inputdatapg, text="INPUT DATA", font=("Comic Sans MS", 25, "bold"))
inputdatalb.grid(row=0, column=0, padx=20, pady=20)
# About Page Frame
waterreportpg = tk.Frame(mainFrame, bg="#F1F1F1")
waterreportlb = tk.Label(waterreportpg, text="WATER QUALITY REPORT", font=("Comic Sans MS", 25, "bold"))
waterreportlb.grid(row=0, column=0, padx=20, pady=20)
# Prediction Tool Page Frame
predictiontoolpg = tk.Frame(mainFrame, bg="#F1F1F1")
predictiontoollb = tk.Label(predictiontoolpg, text="PREDICTION TOOLS", font=("Comic Sans MS", 25, "bold"))
predictiontoollb.grid(row=0, column=0, padx=20, pady=20)    
# Prediction Tool Page Frame
settingspg = tk.Frame(mainFrame, bg="#F1F1F1")
settingslb = tk.Label(settingspg, text="SETTINGS", font=("Comic Sans MS", 25, "bold"))
settingslb.grid(row=0, column=0, padx=20, pady=20)


# Load icons
DBIcon = ImageTk.PhotoImage(Image.open('Icons/DBIcon.png').resize((25,25)))
IPIcon = ImageTk.PhotoImage(Image.open('Icons/IPIcon.png').resize((25,25)))
WQRIcon = ImageTk.PhotoImage(Image.open('Icons/WQRIcon.png').resize((25,25)))
PTIcon = ImageTk.PhotoImage(Image.open('Icons/PTIcon.png').resize((25,25)))
SIcon = ImageTk.PhotoImage(Image.open('Icons/SIcon.png').resize((25,25)))

# Define buttons
bg1 = tk.Button(navBar, text="L", width=5, height=2, relief="flat", bg="#1d97bd", fg="#F1F1F1", command=lambda: indicate(bg1,about_page))
bg2 = tk.Button(navBar, image=DBIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: indicate(bg2,dashboard_page))
bg3 = tk.Button(navBar, image=IPIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: indicate(bg3,inputdata_page))
bg4 = tk.Button(navBar, image=WQRIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: indicate(bg4,waterreport_page))
bg5 = tk.Button(navBar, image=PTIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: indicate(bg5,predictiontool_page))
bg6 = tk.Button(navBar, image=SIcon, bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: indicate(bg6,settings_page))

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

about_page()

navBar.propagate(False)
root.mainloop()
