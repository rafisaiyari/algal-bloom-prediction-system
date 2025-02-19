import tkinter as tk
from tkinter import ttk
from tkinter import Label, Tk, Entry
from tkcalendar import DateEntry
import pandas as pd
from PIL import Image, ImageTk, ImageDraw
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


import csv

root = tk.Tk()
root.minsize(800, 600)
root.geometry('1280x720')

# Load icons
AboutIcon = ImageTk.PhotoImage(Image.open('Icons/AppLogo.png').resize((200,65)))
AppIcon = ImageTk.PhotoImage(Image.open('Icons/AppIcon.png').resize((35,35)))
AppLogo = ImageTk.PhotoImage(Image.open('Icons/AppLogo.png').resize((120,40)))

DBIcon = ImageTk.PhotoImage(Image.open('Icons/DBIcon.png').resize((25,25)))
IPIcon = ImageTk.PhotoImage(Image.open('Icons/IPIcon.png').resize((25,25)))
WQRIcon = ImageTk.PhotoImage(Image.open('Icons/WQRIcon.png').resize((25,25)))
PTIcon = ImageTk.PhotoImage(Image.open('Icons/PTIcon.png').resize((25,25)))
SIcon = ImageTk.PhotoImage(Image.open('Icons/SIcon.png').resize((25,25)))



min_w = 85   # Minimum width of the navBar
max_w = 200  # Maximum width of the navBar
cur_width = min_w  # Initial width
expanded = False  # Expansion flag

is_hovering = False  # Track mouse hover state


# Sidebar Frame --------------------------------------------------------------------------------------
navBar = tk.Frame(root, width=cur_width, height=root.winfo_height(), bg="#1d97bd")
navBar.grid(row=0, column=0, sticky="ns")

mainFrame = tk.Frame(root, width=(root.winfo_width()-100), height=root.winfo_height(), bg="#F1F1F1")
mainFrame.grid(row=0, column=1, sticky="nsew")

# Define buttons
btn1 = tk.Button(navBar, image=AppIcon, cursor="hand2", anchor="center", bg="#F1F1F1", relief="flat" , fg="#F1F1F1", command=lambda: call_page(btn1,about_page))
btn2 = tk.Button(navBar, image=DBIcon, cursor="hand2", anchor="center", bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: call_page(btn2,dashboard_page))
btn3 = tk.Button(navBar, image=IPIcon, cursor="hand2", anchor="center", bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: call_page(btn3,inputdata_page))
btn4 = tk.Button(navBar, image=WQRIcon, cursor="hand2", anchor="center", bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: call_page(btn4,waterreport_page))
btn5 = tk.Button(navBar, image=PTIcon, cursor="hand2", anchor="center", bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: call_page(btn5,predictiontool_page))
btn6 = tk.Button(navBar, image=SIcon, cursor="hand2", anchor="center", bg="#1d97bd", relief="flat", fg="#F1F1F1", command=lambda: call_page(btn6,settings_page))

# Function to expand navbar
def expand():
    global cur_width, expanded
    cur_width += 5
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
    cur_width -= 5
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
        btn1.config(text="", image=AppLogo, width=20, bg="#FFFFFF")
        btn2.config(text=" DASHBOARD", image=DBIcon, compound="left", fg="#F1F1F1", width=20, bg="#1d97bd", anchor="w", padx=10)
        btn3.config(text=" INPUT DATA", image=IPIcon, compound="left", fg="#F1F1F1", width=20, bg="#1d97bd", anchor="w", padx=10)
        btn4.config(text=" WATER QUALITY\nREPORT", image=WQRIcon, compound="left", fg="#F1F1F1", width=20, height=30, bg="#1d97bd", anchor="w", padx=10)
        btn5.config(text=" PREDICTION TOOL", image=PTIcon, compound="left", fg="#F1F1F1", width=20, bg="#1d97bd", anchor="w", padx=10)
        btn6.config(text=" SETTINGS", image=SIcon, compound="left", fg="#F1F1F1", width=20, bg="#1d97bd", anchor="w", padx=10)
    else:
        btn1.config(text="", image=AppIcon, anchor="center")
        btn2.config(text="", image=DBIcon, width=10, anchor="center")
        btn3.config(text="", image=IPIcon, width=10, anchor="center")
        btn4.config(text="", image=WQRIcon, width=10, anchor="center")
        btn5.config(text="", image=PTIcon, width=10, anchor="center")
        btn6.config(text="", image=SIcon, width=10, anchor="center")

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
    
def update_size(event=None):
    navBar.config(height=root.winfo_height())
    mainFrame.config(height=root.winfo_height(), width=(root.winfo_width()))

def reset_indicator():
    btn1.config(relief="flat")
    btn2.config(relief="flat")
    btn3.config(relief="flat")
    btn4.config(relief="flat")
    btn5.config(relief="flat")
    btn6.config(relief="flat")
    
def delete_page():
    for frame in mainFrame.winfo_children():
        frame.grid_forget()

def call_page(btn, page):
    reset_indicator()  # Call reset
    if (btn!=None):
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
    load_csv_data()
    create_legend()

def predictiontool_page():
    # Ensure the page appears with grid settings
    predictiontoolpg.grid(row=0, column=1, sticky="nsew")

def settings_page():
    # Ensure the page appears with grid settings
    settingspg.grid(row=0, column=1, sticky="nsew")

def create_legend():
    # Dictionary to hold legend information for each parameter
    legend_data = {
        "pH": [
            {"color": "blue", "label": "Conformed with Classes A and B (6.5-8.5)"},
            {"color": "lightblue", "label": "Conformed with Class C (6.5-9.0)"},
            {"color": "green", "label": "Conformed with Class D (6.0-9.0)"},
            {"color": "red", "label": "Failed Guidelines (< 6 or > 9)"},
            {"color": "gray", "label": "No data"},
        ],
        "Nitrate": [
            {"color": "blue", "label": "Conformed with Classes A, B, C (< 7 mg/L)"},
            {"color": "lightblue", "label": "Conformed with Class D (7-15 mg/L)"},
            {"color": "red", "label": "Failed Guidelines (> 15 mg/L)"},
            {"color": "gray", "label": "No data"},
        ],
        "Ammonia": [
            {"color": "blue", "label": "Conformed with Classes A, B, C (< 0.06 mg/L)"},
            {"color": "lightblue", "label": "Conformed with Class D (0.06-0.30 mg/L)"},
            {"color": "red", "label": "Failed Guidelines (> 0.30 mg/L)"},
            {"color": "gray", "label": "No data"},
        ],
        "Phosphate": [
            {"color": "blue", "label": "Conformed with Classes A, B, C (< 0.025 mg/L)"},
            {"color": "lightblue", "label": "Conformed with Class D (0.025-0.05 mg/L)"},
            {"color": "red", "label": "Failed Guidelines (> 0.05 mg/L)"},
            {"color": "gray", "label": "No data"},
        ],
    }
    # Remove existing legend frames
    for widget in waterreportpg.winfo_children():
        if isinstance(widget, tk.Frame) and widget.winfo_name().startswith("!legend_"):
            widget.destroy()

    row_num = 3  # Start placing legends below the treeview
    for parameter, items in legend_data.items():
        legend_frame = tk.Frame(waterreportpg, bg="#F1F1F1", padx=20, pady=10, name=f"!legend_{parameter}")
        legend_frame.grid(row=row_num, column=0, sticky="w")

        title_label = tk.Label(legend_frame, text=f"{parameter} Legend:", font=("Arial", 12, "bold"), bg="#F1F1F1")
        title_label.pack(side="top", anchor="w")

        for item in items:
            # Create a Canvas widget for the colored rectangle
            color_canvas = tk.Canvas(legend_frame, width=20, height=10, highlightthickness=0, bg="#F1F1F1") # Adjust width and height as needed
            color_canvas.create_rectangle(2, 2, 18, 8, fill=item["color"], outline="")  # Create the rectangle, adjust coordinates for border if needed.
            color_canvas.pack(side="left", padx=(0, 5))

            text_label = tk.Label(legend_frame, text=item["label"], bg="#F1F1F1")
            text_label.pack(side="left")

        row_num += 1

def  load_csv_data():
    df= pd.read_csv("CSV\\Station_1_CWB.csv")    
    df = df.fillna("")

    tree.delete(*tree.get_children())

    for index, row in df.iterrows():
            row_values = list(row)  
            tree.insert("", "end", values=row_values)

def submitData():
    collected_data = []
    for station, headers in entries.items():
        row_data = [station]
        row_data.extend([entry.get() for entry in headers.values()])
        collected_data.append(row_data)

    with open("Sample_Data.csv", "w", newline = "") as file:
        writer = csv.writer(file)

        writer.writerow(["Station", "pH", "Ammonia", "Nitrate", "Phosphate"])
        writer.writerows(collected_data)

    print("Data saved!")
    print (collected_data)

def make_circle_image(image_path, size=(100, 100)):
    img = Image.open(image_path).resize(size, Image.LANCZOS)  # Resize image
    mask = Image.new("L", size, 0)  # Create a blank mask
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size[0], size[1]), fill=255)  # Draw a white circle
    img.putalpha(mask)  # Apply mask
    return ImageTk.PhotoImage(img)

P1 = make_circle_image("DevPics/Benj.jpg", size=(150,150))
P2 = make_circle_image("DevPics/Matt.jpg", size=(150,150))
P3 = make_circle_image("DevPics/Rafi.jpg", size=(150,150))
P4 = make_circle_image("DevPics/Beau.jpg", size=(150,150))


# Bind resize event
root.bind("<Configure>", update_size)



# About Page Frame --------------------------------------------------------------------------------------
aboutpg = tk.Frame(mainFrame, bg="#F1F1F1")
aboutlb = tk.Label(aboutpg, text="ABOUT US: THE DEVELOPERS", justify="left", anchor="w", font=("Arial", 25, "bold"))
aboutlb.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="ew")
aboutpg.rowconfigure(0,weight=1)
aboutpg.columnconfigure(0,weight=1) 
 
 
 
DP1 = tk.Label(aboutpg, image=P1, anchor="center")
DP1.grid(row=1, column=0, padx=50, pady=50, sticky="ew")
N1 = tk.Label(aboutpg, text="Franz Benjamin Africano", anchor="center", font=("Arial", 11, "bold"))
N1.grid(row=2, column=0, pady=10, sticky="ew")

DP2 = tk.Label(aboutpg, image=P2, anchor="center")
DP2.grid(row=1, column=1, padx=50, pady=50, sticky="ew")
N2 = tk.Label(aboutpg, text="Matt Terrence Rias", anchor="center", font=("Arial", 11, "bold"))
N2.grid(row=2, column=1, pady=10, sticky="ew")

DP3 = tk.Label(aboutpg, image=P3, anchor="center")
DP3.grid(row=1, column=2, padx=50, pady=50, sticky="ew")
N3 = tk.Label(aboutpg, text="Rafi Saiyari", anchor="center", font=("Arial", 11, "bold"))
N3.grid(row=2, column=2, pady=10, sticky="ew")

DP4 = tk.Label(aboutpg, image=P4, anchor="center")
DP4.grid(row=1, column=3, padx=50, pady=50, sticky="ew")
N4 = tk.Label(aboutpg, text="Beau Lawyjet Sison", anchor="center", font=("Arial", 11, "bold"))
N4.grid(row=2, column=3, pady=10, sticky="ew")

# Project Information
project_info = (
    "We are a team of dedicated researchers and software engineers from the Bachelor of Science in Computer Science with specialization in Software Engineering program currently studying in FEU Institute of Technology. Our project, 'Prediction of Algal Blooms using Long Short-Term Memory Algorithm in Laguna Lake,' aims to create an advanced predictive system that monitors and forecasts harmful algal blooms, assisting communities, researchers, and local government units in managing and mitigating its impact."
)

project_label = tk.Label(aboutpg, text=project_info, wraplength=800, justify="center", bg="#F1F1F1", font=("Arial", 12))
project_label.grid(row=3, column=0, columnspan=4, pady=50,padx=20, sticky="ew")

# Dashboard Page Frame --------------------------------------------------------------------------------------
dashboardpg = tk.Frame(mainFrame, bg="#F1F1F1")
dashboardlb = tk.Label(dashboardpg, text="DASHBOARD", font=("Arial", 25, "bold"))
dashboardlb.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

# Sample Data 
years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
nitrate = [0.331, 0.308, 0.265, 0.121, 0.223, 0.405, 0.267, 0.413]

# Create Matplotlib Figure for line graph
fig1 = Figure(figsize=(5, 2), dpi=100)
fig1.patch.set_color("#F1F1F1")  # Transparent figure background
plot1 = fig1.add_subplot(111)
plot1.plot(years, nitrate, marker='o', linestyle='-', color='teal', label="Nitrate")
plot1.set_frame_on(False)
plot1.set_title("Yearly Nitrate Levels")
plot1.set_xlabel("Year")
plot1.set_ylabel("Nitrate")
plot1.set_facecolor("None")  # Transparent plot background
plot1.set_ylim(0.000, 0.800)
plot1.legend()

# Create Matplotlib Figure for bar graph
fig2 = Figure(figsize=(5, 2), dpi=100)
fig2.patch.set_color("#F1F1F1")
plot2 = fig2.add_subplot(111)  # Create a new subplot for the bar graph
plot2.bar(years, nitrate, color='blue', label="Nitrate", width=0.8)
plot2.set_frame_on(False)
plot2.set_title("Yearly Nitrate Levels")
plot2.set_xlabel("Year")
plot2.set_ylabel("Nitrate Level")
plot2.set_ylim(0.000, 0.800)
plot2.legend()
plot2.set_facecolor("#F1F1F1")  # Make the plot background transparent

# Embed Matplotlib in Tkinter
canvas1 = FigureCanvasTkAgg(fig1, master=dashboardpg)
canvas1.draw()
canvas1.get_tk_widget().grid(row=1, column=0, padx=30, pady=30)

# Embed Matplotlib in Tkinter for the bar graph
canvas2 = FigureCanvasTkAgg(fig2, master=dashboardpg)
canvas2.draw()
canvas2.get_tk_widget().grid(row=2, column=0, padx=30, pady=30)

# Input Data Page Frame --------------------------------------------------------------------------------------
inputdatapg = tk.Frame(mainFrame, bg="#F1F1F1")
inputdatalb = tk.Label(inputdatapg, text="INPUT DATA", font=("Arial", 25, "bold"))
inputdatalb.grid(row=0, column=0, padx=20, pady=20)

headers = ["pH", "Ammonia", "Nitrate", "Phosphate"]
tk.Label(inputdatapg, text="Station").grid(column=0, row=1, padx=5, pady=5)
for col, header in enumerate(headers, start=1):
    tk.Label(inputdatapg, text=header).grid(column=col, row=1, padx=5, pady=5)

stations = ["I", "II", "IV", "V", "VII", "XV", "XVI", "XVII", "XVIII"]

entries = {}

for row, station in enumerate(stations, start=2):
    tk.Label(inputdatapg, text=f"Station {station}:").grid(column=0, row=row, padx=5, pady=5)

    entries[station] = {}  # Store entries in a nested dictionary

    for col, header in enumerate(headers, start=1):
        entry = tk.Entry(inputdatapg)
        entry.grid(column=col, row=row, padx=5, pady=5)
        entries[station][header] = entry  # Store entry widget correctly

yearInput=DateEntry(inputdatapg,selectmode='year', width=10)
yearInput.grid(column=0, row=12)

submit_button = tk.Button(inputdatapg, text="Submit", command=submitData)
submit_button.grid(column=0, row=len(stations) + 3, columnspan=5, pady=10)

# Water Quality Report Page Frame --------------------------------------------------------------------------------------
waterreportpg = tk.Frame(mainFrame, bg="#F1F1F1")
waterreportlb = tk.Label(waterreportpg, text="WATER QUALITY REPORT", font=("Segoe UI", 25, "bold"))
waterreportlb.grid(row=0, column=0, padx=20, pady=20, sticky="w")

tree =ttk.Treeview(waterreportpg, height = 30)
tree.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

df= pd.read_csv("CSV\\Station_1_CWB.csv")
tree["columns"] = list(df.columns)
tree["show"] = "headings"

for col in df.columns:
    tree.heading(col, text=col)
    tree.column(col, width=120, anchor="center")

load_csv_data()
waterreport_page()

# Prediction Tool Page Frame --------------------------------------------------------------------------------------
predictiontoolpg = tk.Frame(mainFrame, bg="#F1F1F1")
predictiontoollb = tk.Label(predictiontoolpg, text="PREDICTION TOOLS", font=("Arial", 25, "bold"))
predictiontoollb.grid(row=0, column=0, padx=20, pady=20) 
   
# Settings Tool Page Frame --------------------------------------------------------------------------------------
settingspg = tk.Frame(mainFrame, bg="#F1F1F1")
settingslb = tk.Label(settingspg, text="SETTINGS", font=("Arial", 25, "bold"))
settingslb.grid(row=0, column=0, padx=20, pady=20)



# Pack buttons with equal width
btn1.pack(fill="x", padx=20, pady=20)
btn2.pack(fill="x", padx=20, pady=10)
btn3.pack(fill="x", padx=20, pady=10)
btn4.pack(fill="x", padx=20, pady=10)
btn5.pack(fill="x", padx=20, pady=10)
btn6.pack(side="bottom", fill="x", padx=20, pady=25)

# Bind hover events
navBar.bind('<Enter>', on_enter)
navBar.bind('<Leave>', on_leave)

for btn in [btn1, btn2, btn3, btn4, btn5, btn6]:
    btn.bind('<Enter>', on_enter)
    btn.bind('<Leave>', on_leave)


call_page(None, dashboard_page)

navBar.propagate(False)
mainFrame.propagate(False)

root.mainloop()
