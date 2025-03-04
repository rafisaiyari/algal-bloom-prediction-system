import tkinter as tk
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DashboardPage(tk.Frame):
    def __init__(self, parent, bg="#F1F1F1"):
        super().__init__(parent, bg=bg)
        self.parent = parent
        self.create_widgets()

    def create_widgets(self):
        dashboardlb = tk.Label(self, text="DASHBOARD", font=("Arial", 25, "bold"))
        dashboardlb.grid(row=0, column=0, padx=20, pady=20, sticky="nw", columnspan=4)

        def load_csv_data_for_dash(filename):
            try:
                df = pd.read_csv(filename)  # Load CSV file

                # Convert 'Date' column to datetime format
                df["Date"] = pd.to_datetime(df["Date"], format="%b %Y", errors='coerce')
                df["Year"] = df["Date"].dt.year
                df["Month"] = df["Date"].dt.month

                # Convert numeric columns
                for col in ["pH", "Ammonia", "Nitrate", "Phosphate"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                return df  # Return full dataframe
            except Exception as e:
                print("Error loading CSV:", e)
                return pd.DataFrame()

        # Load Data
        self.csv_file = "CSV/Station_1_CWB.csv"
        self.df = load_csv_data_for_dash(self.csv_file)

        # Dropdown to select year
        self.year_label = tk.Label(self, text="Select Year:", font=("Arial", 12))
        self.year_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.year_var = tk.StringVar()
        self.year_var.set("2016")  # Default year

        self.year_dropdown = tk.OptionMenu(self, self.year_var, *[str(y) for y in range(2016, 2026)])
        self.year_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Radio buttons for parameters
        self.param_var = tk.StringVar(value="pH")  # Default selection
        self.param_buttons = []

        self.param_var_cb = {param: tk.IntVar() for param in
                             ["pH", "Ammonia", "Nitrate", "Phosphate", "Chlorophyll-a", "Temperature", "DO"]}
        self.param_checkboxes = []

        self.mode = "yearly"
        self.b = tk.Button()

        def disp_btn():
            # Clear existing buttons before adding new ones
            for btn in self.param_checkboxes:
                btn.grid_forget()
            self.param_checkboxes.clear()

            for btn in self.param_buttons:
                btn.grid_forget()
            self.param_buttons.clear()

            if self.mode == "yearly":
                for i, param in enumerate(self.param_var_cb.keys()):
                    cb = tk.Checkbutton(self, text=param, variable=self.param_var_cb[param], font=("Arial", 10))
                    cb.grid(row=2, column=i, padx=5, pady=5, sticky="w")
                    self.param_checkboxes.append(cb)

            elif self.mode == "monthly":
                for i, param in enumerate(
                        ["pH", "Ammonia", "Nitrate", "Phosphate", "Chlorophyll-a", "Temperature", "DO"]):
                    rb = tk.Radiobutton(self, text=param, variable=self.param_var, value=param, font=("Arial", 10))
                    rb.grid(row=2, column=i, padx=5, pady=5, sticky="w")
                    self.param_buttons.append(rb)

        disp_btn()

        self.error_message = tk.Label(self, text="No Parameter Selected.", font=("Arial", 12))

        def display_selected_data_bar():
            self.mode = "monthly"
            disp_btn()
            self.error_message.grid_forget()
            self.year_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
            self.year_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky="w")
            selected_year = int(self.year_var.get())
            selected_param = self.param_var.get()

            if not selected_param:
                self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                return

            filtered_df = self.df[self.df["Year"] == selected_year]

            if filtered_df.empty:
                print(f"No data available for the year {selected_year}!")
                self.error_message.config(text=f"No data available for {selected_year}.")
                self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                return

            # Create bar graph
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            months = range(1, 13)  # Months from 1 to 12

            monthly_avg = filtered_df.groupby("Month")[selected_param].mean()

            if not monthly_avg.empty and not monthly_avg.isnull().all():
                ax.bar(months, monthly_avg, label=selected_param, alpha=0.7)  # Bar chart
                ax.set_title(f"Monthly Data for {selected_year}")
                ax.set_xlabel("Month")
                ax.set_ylabel("Value")
                ax.set_xticks(months)
                ax.legend()
                ax.grid(True)
            else:
                print("No valid data available for the selected parameter.")
                self.error_message.config(text="No valid data available for selected parameter.")
                self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                return

            # Update canvas (Clear the old plot)
            self.canvas1.get_tk_widget().destroy()
            self.canvas1 = FigureCanvasTkAgg(fig, master=self)
            self.canvas1.draw()
            self.canvas1.get_tk_widget().grid(row=3, column=0, columnspan=6, padx=30, pady=30)

        def display_selected_data_line():
            self.mode = "yearly"
            disp_btn()
            self.error_message.grid_forget()
            self.year_label.grid_forget()
            self.year_dropdown.grid_forget()

            # Get selected parameters from checkboxes
            selected_params = [param for param, var in self.param_var_cb.items() if var.get() == 1]

            if not selected_params:
                self.error_message.config(text="No Parameter Selected.")
                self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=4)
                return

            # Group data by Year and calculate the mean for selected parameters
            filtered_df = self.df.groupby("Year")[selected_params].mean().dropna()

            if filtered_df.empty:
                print("No data available for the selected parameters!")
                self.error_message.config(text="No data available for selected parameters.")
                self.error_message.grid(row=4, column=0, padx=20, pady=20, sticky="nw", columnspan=6)
                return

            # Create the plot
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

            # Plot multiple parameters
            for param in selected_params:
                ax.plot(filtered_df.index, filtered_df[param], marker='o', linestyle='-', label=param)

            ax.set_title("Yearly Data for Selected Parameters")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)

            # Update canvas (Clear old plot)
            self.canvas1.get_tk_widget().destroy()
            self.canvas1 = FigureCanvasTkAgg(fig, master=self)
            self.canvas1.draw()
            self.canvas1.get_tk_widget().grid(row=3, column=0, columnspan=4, padx=30, pady=30)

        # Button to update graph
        monthly_btn = tk.Button(self, text="Get Monthly Data", command=display_selected_data_bar, font=("Arial", 12),
                                bg="#1d97bd", fg="white")
        monthly_btn.grid(row=5, column=0, padx=10, pady=5, sticky="w")

        yearly_btn = tk.Button(self, text=" Get Yearly Data", command=display_selected_data_line, font=("Arial", 12),
                               bg="#1d97bd", fg="white")
        yearly_btn.grid(row=5, column=1, padx=10, pady=5, sticky="w")

        # Initial yearly average line graph
        def plot_yearly_average():
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)

            if not self.df.empty:
                yearly_avg = self.df.groupby("Year")[["Ammonia", "Nitrate", "Phosphate"]].mean()

                for param in yearly_avg.columns:
                    ax.plot(yearly_avg.index, yearly_avg[param], marker='o', linestyle='-', label=param)

                ax.set_title("Yearly Average of Water Quality Parameters")
                ax.set_xlabel("Year")
                ax.set_ylabel("Average Value")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No Data Available", fontsize=14, ha='center', va='center')

            return fig

        self.canvas1 = FigureCanvasTkAgg(plot_yearly_average(), master=self)
        self.canvas1.get_tk_widget().grid(row=3, column=0, columnspan=4, padx=30, pady=30)

    def show(self):
        self.grid(row=0, column=0, sticky="nsew")