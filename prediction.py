import customtkinter as ctk
import webview
import os
import pandas as pd
import folium
from folium import plugins
import geopandas as gpd
from Heatmapper import HeatmapByParameter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Import OpenMeteo libraries
import openmeteo_requests
import requests_cache
from retry_requests import retry


class PredictionPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="#FFFFFF")

        # File paths and coordinates
        self.geojson_path = "heatmapper/stations_final.geojson"
        self.laguna_coords = {"lat": 14.35, "lon": 121.2}
        self.station_coords = []

        # Add extreme values toggle flag
        self.use_extreme_values = ctk.BooleanVar(value=False)

        self.data = pd.read_csv("CSV/chlorophyll_predictions_by_station.csv")
        self.data["Date"] = pd.to_datetime(self.data["Date"])

        self.data = self.data.dropna(subset=['Date'])

        self.data["Year"] = self.data["Date"].dt.year
        self.data["Month"] = self.data["Date"].dt.month_name()

        # Load station coordinates on initialization
        self.load_station_coordinates()

        self.title_label = ctk.CTkLabel(
            self,
            text="PREDICTION PAGE",
            font=("Segoe UI", 25, "bold")
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        # Create top control frame with fixed height
        self.control_frame = ctk.CTkFrame(
            self,
            height=200,
            fg_color="#1f6aa5",
            border_width=0  # Remove shadow/border
        )
        self.columnconfigure(0, weight=1)  # Allow the control frame to expand
        self.rowconfigure(1, weight=1)
        self.control_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="nsew")
        self.control_frame.grid_rowconfigure(0, weight=1)

        self.controls_container = ctk.CTkFrame(
            self.control_frame,
            fg_color="#FFFFFF",  # White background
            border_width=0
        )
        self.controls_container.pack(pady=10, padx=20, anchor="center")

        # Add month selection controls
        self.year_label = ctk.CTkLabel(
            self.controls_container,
            text="Year :",
            font=("Segoe UI", 14)
        )
        self.year_label.grid(row=0, column=0, padx=(0, 5), pady=5)

        # Year Dropdown
        self.year_var = ctk.StringVar(value=str(self.data["Year"].iloc[0]))
        year_values = [str(year) for year in self.data["Date"].dt.year.unique()]
        self.year_dropdown = ctk.CTkOptionMenu(
            self.controls_container,
            values=year_values,
            variable=self.year_var,
            width=100,
            height=30,
            font=("Segoe UI", 14),
            fg_color="#1f6aa5",

        )
        self.year_dropdown.grid(row=0, column=1, padx=5, pady=5)

        self.month_label = ctk.CTkLabel(
            self.controls_container,
            text="Month :",
            font=("Segoe UI", 14)
        )
        self.month_label.grid(row=0, column=2, padx=(10, 5), pady=5)

        month_values = self.data["Month"].dropna().unique().tolist()

        if isinstance(month_values[0], str):
            month_values.sort()

            month_order = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                           "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}

            # Sort by actual month order
            month_values.sort(key=lambda x: month_order.get(x, 13))

            # Set January as the default
            self.month_var = ctk.StringVar(value="January")

        elif all(isinstance(x, (int, float)) for x in month_values):
            # Sort numerically
            month_values.sort()

            # Set January (1) as the default
            self.month_var = ctk.StringVar(value=str(1))  # or value=1 depending on what your code expects

        # Create the dropdown
        self.month_dropdown = ctk.CTkOptionMenu(
            self.controls_container,
            values=[str(month) for month in month_values],  # Convert all to strings for display
            variable=self.month_var,
            width=100,
            height=30,
            font=("Segoe UI", 14),
            fg_color="#1f6aa5",
        )
        self.month_dropdown.grid(row=0, column=3, padx=5, pady=5)

        self.button_frame = ctk.CTkFrame(
            self.controls_container,
            fg_color="transparent"  # Make it invisible
        )
        self.button_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky="ew")

        # Configure button frame columns to be equal width
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=1)
        self.button_frame.columnconfigure(2, weight=1)

        self.combined_map_button = ctk.CTkButton(
            self.button_frame,
            text="Combined Map",
            width=120,
            height=30,
            fg_color="#1f6aa5",
            hover_color="#18558a",
            command=self.show_combined_map
        )
        self.combined_map_button.grid(row=0, column=0, padx=10, pady=0, sticky="ew")

        self.preview_button = ctk.CTkButton(
            self.button_frame,
            text="Preview Data",
            width=150,
            height=30,
            fg_color="#1f6aa5",
            hover_color="#18558a",
            command=self.show_chlorophyll_preview
        )
        self.preview_button.grid(row=0, column=1, padx=10, pady=0, sticky="ew")

        self.run_button = ctk.CTkButton(
            self.button_frame,
            text="Run Model",
            width=150,
            height=30,
            fg_color="#1f6aa5",
            hover_color="#18558a",
            command=self.run_chlorophyll_model
        )
        self.run_button.grid(row=0, column=2, padx=10, pady=0, sticky="ew")

        # Set up the trace for dropdown changes
        self.year_var.trace_add("write", self.update_data_selection)
        self.month_var.trace_add("write", self.update_data_selection)

        # Change control frame and content frame background to white
        self.control_frame.configure(fg_color="#FFFFFF")
        self.content_frame = ctk.CTkFrame(self.control_frame, fg_color="#FFFFFF", )
        self.content_frame.pack(fill="both", expand=True)

    def update_data_selection(self, *args):
        """Update filtered data when year or month selection changes"""
        try:
            selected_year = int(self.year_var.get())
            selected_month = self.month_var.get()

            print(f"Selection changed to {selected_month} {selected_year}")

            # Update UI for the current selection
            # If there's a plot already displayed, update it
            if len(self.content_frame.winfo_children()) > 0:
                self.show_chlorophyll_preview()
        except Exception as e:
            print(f"Error updating selection: {e}")

    def load_station_coordinates(self):
        """Load station coordinates from GeoJSON file"""
        try:
            # Check if file exists
            if not os.path.exists(self.geojson_path):
                print(f"Error: Station GeoJSON file not found: {self.geojson_path}")
                self.station_coords = []
                return

            # Read GeoJSON using geopandas
            gdf = gpd.read_file(self.geojson_path)

            # Extract coordinates from each station
            self.station_coords = []
            for _, row in gdf.iterrows():
                geom = row.geometry
                station_id = str(row["id"])

                # Extract point coordinates
                if geom.geom_type == 'MultiPoint':
                    point = geom.geoms[0]
                elif geom.geom_type == 'Point':
                    point = geom
                else:
                    print(f"Skipping non-point geometry: {geom.geom_type}")
                    continue

                # Store station data with coordinates
                self.station_coords.append({
                    "id": station_id,
                    "lat": point.y,
                    "lon": point.x,
                    "name": row.get("name", f"Station {station_id}")
                })

            print(f"Loaded {len(self.station_coords)} station coordinates")

        except Exception as e:
            print(f"Error loading GeoJSON file {self.geojson_path}: {e}")
            import traceback
            traceback.print_exc()
            self.station_coords = []

    def show_combined_map(self):
        """Show a combined map with station heatmap and direction data"""
        try:
            # Create directory for output if it doesn't exist
            os.makedirs("combined_maps", exist_ok=True)

            # Path to lake boundary file
            lake_boundary_path = "heatmapper/laguna_lakeFinal.geojson"

            # Verify file paths exist
            if not os.path.exists(self.geojson_path):
                print(f"Error: Station GeoJSON file not found: {self.geojson_path}")
                return

            if not os.path.exists(lake_boundary_path):
                print(f"Warning: Lake boundary file not found: {lake_boundary_path}")

            # Setup the Open-Meteo API client with cache and retry
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)

            # Dictionary to store direction data by station ID
            wind_data_d = {}
            wind_data_s = {}

            # Collect wind direction data for each station
            print("Collecting wind direction data...")
            for station in self.station_coords:
                station_id = station['id']

                # Set parameters for API request
                params = {
                    "latitude": station["lat"],
                    "longitude": station["lon"],
                    "daily": ["wind_speed_10m_max", "wind_direction_10m_dominant"],
                    "timezone": "Asia/Singapore",
                    "forecast_days": 16
                }

                try:

                    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
                    response = responses[0]

                    # Process daily data
                    daily = response.Daily()
                    wind_direction = int(daily.Variables(1).ValuesAsNumpy()[0])

                    # Get wind speed (keep it in km/h)
                    wind_speed = float(daily.Variables(0).ValuesAsNumpy()[0])

                    # Verify that the unit is km/h
                    daily_variables = daily.Variables(1)  # Wind speed variable
                    variable_unit = daily_variables.Unit()

                    print(f"Wind speed unit from API: {variable_unit}")

                    # In case the API returns a different unit, handle conversion
                    if variable_unit == "m/s":
                        # Convert from m/s to km/h
                        wind_speed = wind_speed * 3.6
                        print(f"Converted wind speed from m/s to km/h: {wind_speed} km/h")

                    # Additional safety cap - cap at 150 km/h as reasonable max
                    if wind_speed > 150:
                        print(f"Warning: Capping extremely high wind speed from {wind_speed} to 150 km/h")
                        wind_speed = 150.0

                    # Store wind direction and speed in separate dictionaries
                    wind_data_d[station_id] = wind_direction
                    wind_data_s[station_id] = wind_speed

                    # Debug output
                    print(f"Station {station_id}: Wind Direction={wind_direction}°, Wind Speed={wind_speed} km/h")

                except Exception as e:
                    print(f"Error getting wind data for station {station_id}: {e}")

            # Get selected month and year
            selected_month = self.month_var.get()
            selected_year = int(self.year_var.get())

            # Improved filtering with diagnostics
            print(f"Filtering data for {selected_month} {selected_year}")
            print(f"Data shape before filtering: {self.data.shape}")

            # Filter by year correctly (making sure types match)
            filtered_year = self.data[self.data["Year"] == selected_year]
            print(f"After year filtering: {filtered_year.shape} rows remain")

            # Filter by month (case-insensitive matching for robustness)
            filtered_month = filtered_year[filtered_year["Month"].str.lower() == selected_month.lower()]
            print(f"After month filtering: {filtered_month.shape} rows remain")

            # Don't drop all NaN rows, just those with NaN in critical columns
            filtered_values = filtered_month.copy()

            # Debug: print columns and first few rows
            print(f"Available columns: {filtered_values.columns.tolist()}")
            if not filtered_values.empty:
                print("First few rows of filtered data:")
                print(filtered_values.head().to_string())

                # Check for chlorophyll data specifically
                if "Chlorophyll-a (ug/L)" in filtered_values.columns:
                    chloro_count = filtered_values["Chlorophyll-a (ug/L)"].notna().sum()
                    print(f"Rows with valid chlorophyll data: {chloro_count}")

            # Create combined map with both station markers and wind direction
            selected_month = self.month_var.get()
            selected_year = int(self.year_var.get())

            # Create combined map with both station markers and wind direction
            heatmap = HeatmapByParameter()
            output_path = heatmap.create_combined_map(
                self.geojson_path,
                filtered_values,
                lake_boundary_path,
                wind_data_d,
                wind_data_s,
                selected_year,
                selected_month
            )

            if output_path and os.path.exists(output_path):
                # Open in webview
                webview.create_window(
                    "Laguna Lake Combined Map",
                    output_path,
                    width=900,
                    height=700
                )
                webview.start(debug=False)
            else:
                print("Failed to generate combined map or output file not found")

        except Exception as e:
            print(f"Error showing combined map: {e}")
            import traceback
            traceback.print_exc()

    def generate_geoheatmap(self):
        """Generate Folium heatmap based on station data"""
        try:
            # Verify stations are loaded
            if not self.station_coords:
                print("No station coordinates loaded. Attempting to reload...")
                self.load_station_coordinates()
                if not self.station_coords:
                    print("Still no station coordinates available. Cannot proceed.")
                    return

            # Use loaded station coordinates
            heat_data = [[station["lat"], station["lon"]] for station in self.station_coords]

            # Create base map centered on Laguna Lake
            m = folium.Map(
                location=[14.35, 121.2],
                zoom_start=11,
                tiles='CartoDB positron'
            )

            # Add heatmap layer
            plugins.HeatMap(heat_data).add_to(m)

            # Add station markers
            for station in self.station_coords:
                folium.CircleMarker(
                    location=[station["lat"], station["lon"]],
                    radius=5,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.7,
                    popup=f"Station: {station.get('name', station['id'])}"
                ).add_to(m)

            # Create directory if it doesn't exist
            os.makedirs("heatmaps", exist_ok=True)

            # Save map
            heatmap_path = "heatmaps/geoheatmap.html"
            m.save(heatmap_path)

            # Display in webview
            if os.path.exists(heatmap_path):
                webview.create_window(
                    "Laguna Lake Heatmap",
                    heatmap_path,
                    width=900,
                    height=700
                )
                webview.start(debug=False)
            else:
                print(f"Generated map not found at {heatmap_path}")

        except Exception as e:
            print(f"Error generating heatmap: {e}")
            import traceback
            traceback.print_exc()

    def show_chlorophyll_preview(self):
        try:
            # Get selected year and month
            selected_year = int(self.year_var.get())
            selected_month = self.month_var.get()

            print(f"Generating chlorophyll preview for {selected_month} {selected_year}")

            # Filter data by selected year and month, ensuring we include the full month
            filtered_data = self.data[self.data["Year"] == selected_year]
            filtered_data = filtered_data[filtered_data["Month"] == selected_month]

            # Make sure we're not excluding the last day of predictions
            if filtered_data.empty:
                print(f"No data available for {selected_month} {selected_year}")
                self.show_no_data_message()
                return

            # Print debug info about the filtered data
            print(f"Filtered data shape: {filtered_data.shape}")
            print(f"Columns available: {filtered_data.columns.tolist()}")

            # Clear any existing content in the content frame
            for widget in self.content_frame.winfo_children():
                widget.destroy()

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))

            # Extract stations and their corresponding chlorophyll values
            stations = []
            chlorophyll_values = []
            station_names = []

            # Define a mapping of station IDs to more readable names
            station_name_map = {
                "Station_1_CWB": "Central West Bay",
                "Station_2_EastB": "East Bay",
                "Station_4_CentralB": "Central Bay",
                "Station_5_NorthernWestBay": "North West Bay",
                "Station_8_SouthB": "South Bay",
                "Station_15_SanPedro": "San Pedro",
                "Station_16_Sta. Rosa": "Sta. Rosa",
                "Station_17_Sanctuary": "Sanctuary",
                "Station_18_Pagsanjan": "Pagsanjan"
            }

            # Get unique stations in the filtered data
            unique_stations = filtered_data["Station"].unique()

            for station in unique_stations:
                # Get the station's data
                station_data = filtered_data[filtered_data["Station"] == station]

                # Check if we have chlorophyll data for this station
                if "Predicted_Chlorophyll" in station_data.columns and not station_data[
                    "Predicted_Chlorophyll"].isnull().all():
                    # Calculate average chlorophyll for the station (in case of multiple records)
                    avg_chlorophyll = station_data["Predicted_Chlorophyll"].mean()

                    # Add to our lists
                    stations.append(station)
                    chlorophyll_values.append(avg_chlorophyll)

                    # Use the mapping if available, otherwise use station ID
                    display_name = station_name_map.get(station, station)
                    station_names.append(display_name)

            # If we found no data, show message and return
            if not stations:
                print(f"No chlorophyll data available for {selected_month} {selected_year}")
                self.show_no_data_message()
                return

            print(f"Found data for {len(stations)} stations")

            # Set colors based on new chlorophyll value ranges
            colors = []
            for value in chlorophyll_values:
                if value > 150:  # Very High
                    colors.append("#006400")  # Dark Green
                elif value > 100:  # High
                    colors.append("#228B22")  # Forest Green
                elif value > 50:  # Medium
                    colors.append("#3CB371")  # Medium Sea Green
                elif value > 25:  # Low
                    colors.append("#90EE90")  # Light Green
                elif value >= 0:  # Very Low
                    colors.append("#98FB98")  # Pale Green
                else:  # No data or negative values
                    colors.append("#D5DBDB")  # Gray

            # Create bar plot
            bars = ax.bar(station_names, chlorophyll_values, color=colors)

            # Add value labels on top of bars
            for bar, value in zip(bars, chlorophyll_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{value:.2f}',
                        ha='center', va='bottom', fontsize=9)

            # Set title and labels
            ax.set_title(f'Chlorophyll-a Levels by Station ({selected_month} {selected_year})')
            ax.set_xlabel('Station')
            ax.set_ylabel('Chlorophyll-a (μg/L)')

            # Rotate x-tick labels for better readability
            plt.xticks(rotation=45, ha='right')

            # Add grid for easier reading
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Adjust layout to make room for rotated x-tick labels
            plt.tight_layout()

            # Create frame for the chart
            chart_frame = ctk.CTkFrame(self.content_frame)
            chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Embed the plot in the tkinter window
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # Add legend explaining color coding
            legend_frame = ctk.CTkFrame(self.content_frame)
            legend_frame.pack(fill="x", padx=10, pady=(0, 10))

            ctk.CTkLabel(legend_frame, text="Color Legend:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=10)

            # Add color boxes with labels
            legend_items = [
                {"color": "#98FB98", "label": "Very Low (< 25 μg/L)"},
                {"color": "#90EE90", "label": "Low (25-50 μg/L)"},
                {"color": "#3CB371", "label": "Medium (50-100 μg/L)"},
                {"color": "#228B22", "label": "High (100-150 μg/L)"},
                {"color": "#006400", "label": "Very High (> 150 μg/L)"},
                {"color": "#D5DBDB", "label": "No data"}
            ]

            for item in legend_items:
                box = ctk.CTkFrame(legend_frame, width=15, height=15, fg_color=item["color"])
                box.pack(side="left", padx=(10, 5))
                ctk.CTkLabel(legend_frame, text=item["label"]).pack(side="left", padx=(0, 10))

        except Exception as e:
            print(f"Error displaying chlorophyll preview: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_message(str(e))

    def run_chlorophyll_model(self):
        """Run the chlorophyll forecasting model with station ordering"""
        try:
            # Show loading indicator
            for widget in self.content_frame.winfo_children():
                widget.destroy()

            loading_label = ctk.CTkLabel(
                self.content_frame,
                text="Running chlorophyll forecasting model...\nThis may take a few minutes.",
                font=("Segoe UI", 16)
            )
            loading_label.pack(pady=50)
            self.update()

            # Define the standard station order (once)
            station_order = [
                "Station_1_CWB",
                "Station_2_EastB",
                "Station_4_CentralB",
                "Station_5_NorthernWestBay",
                "Station_8_SouthB",
                "Station_15_SanPedro",
                "Station_16_Sta. Rosa",
                "Station_17_Sanctuary",
                "Station_18_Pagsanjan"
            ]

            # Import chlorophyll_forecaster functions
            loading_label.configure(text="Importing chlorophyll forecaster...")
            self.update()

            # Configure matplotlib to not display figures
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            plt.ioff()  # Turn off interactive mode

            # Monkey patch plt.show to prevent figures from popping up
            original_show = plt.show
            plt.show = lambda: None

            from chlorophyll_forecaster import (
                load_existing_model_and_features,
                load_full_dataset,
                retrain_model_on_full_data,
                generate_future_dates,
                prepare_future_features,
                predict_future_values,
                plot_and_save_results
            )

            # Create a patched version of plot_and_save_results that sorts by station
            def patched_plot_and_save_results(df, future_pred_df, target='Chlorophyll-a (ug/L)'):
                """Patched version of plot_and_save_results that sorts by station"""
                import pandas as pd

                # First, extract station names
                station_cols = [col for col in future_pred_df.columns if 'station_' in col]
                if station_cols:
                    # Create a temporary column for station names
                    temp_station_names = []
                    for i, row in future_pred_df.iterrows():
                        station_name = "Unknown"
                        for col in station_cols:
                            if row[col] == 1:
                                parts = col.split('_')
                                if len(parts) > 2:
                                    station_name = '_'.join(parts[2:])
                                    break
                        temp_station_names.append(station_name)

                    # Add a new 'Station' column
                    future_pred_df = future_pred_df.copy()  # Create a copy to avoid modifying the original
                    future_pred_df['Station'] = temp_station_names

                    # Convert to categorical with proper ordering (only with valid stations)
                    valid_stations = [s for s in station_order if s in set(temp_station_names)]
                    future_pred_df["Station"] = pd.Categorical(
                        future_pred_df["Station"],
                        categories=valid_stations,
                        ordered=True
                    )

                    # Sort by Station and Date
                    if "Date" in future_pred_df.columns:
                        future_pred_df = future_pred_df.sort_values(["Station", "Date"])
                    else:
                        future_pred_df = future_pred_df.sort_values("Station")

                # Call the original function with sorted data
                result = plot_and_save_results(df, future_pred_df, target)

                # Close all figures to prevent memory leaks
                plt.close('all')

                return result

            # Set extreme values status based on switch
            extreme_values_status = "enabled" if self.use_extreme_values.get() else "disabled"
            loading_label.configure(text=f"Loading model and features... (Extreme values: {extreme_values_status})")
            self.update()

            # Load the model (always use the same model file)
            model, selected_features, metadata = load_existing_model_and_features()

            # Define features and target
            features = ['pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
                        'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)',
                        'Temperature', 'Phytoplankton']
            target = 'Chlorophyll-a (ug/L)'

            # Load dataset
            loading_label.configure(text="Loading and preprocessing data...")
            self.update()
            file_path = 'CSV/merged_stations.xlsx'
            df, last_date = load_full_dataset(file_path, features, target)

            # Retrain model
            loading_label.configure(text="Retraining model on full dataset...")
            self.update()
            retrained_model, full_r2, full_rmse = retrain_model_on_full_data(
                df, features, target, selected_features, model
            )

            # Get station info
            station_cols = [col for col in df.columns if 'station_' in col]
            unique_station_configs = df[station_cols].drop_duplicates().reset_index(drop=True)
            num_stations = len(unique_station_configs)

            # Generate future dates
            loading_label.configure(text="Generating future dates and features...")
            self.update()
            future_dates_df = generate_future_dates(last_date, months_ahead=18, num_stations=num_stations)

            # Prepare features - THE KEY PART: control extreme value injection with the parameter
            loading_label.configure(text=f"Preparing features with extreme values {extreme_values_status}...")
            self.update()
            future_df = prepare_future_features(
                df, future_dates_df, features, target, selected_features,
                enable_extremity_handling=self.use_extreme_values.get()  # Pass the extremity flag
            )

            # Make predictions
            loading_label.configure(text="Making predictions...")
            self.update()
            future_pred_df = predict_future_values(
                retrained_model,
                future_df,
                selected_features,
                use_log=True
            )

            # Add station names column directly
            loading_label.configure(text="Processing station information...")
            self.update()

            # Create a list of station names
            temp_station_names = []
            for i, row in future_pred_df.iterrows():
                station_name = "Unknown"
                for col in station_cols:
                    if row[col] == 1:
                        parts = col.split('_')
                        if len(parts) > 2:
                            station_name = '_'.join(parts[2:])
                            break
                temp_station_names.append(station_name)

            # Add the Station column directly
            future_pred_df['Station'] = temp_station_names

            # Sort the DataFrame (without making it categorical, to avoid errors)
            station_order_dict = {name: i for i, name in enumerate(station_order)}
            future_pred_df['sort_order'] = future_pred_df['Station'].map(
                lambda x: station_order_dict.get(x, len(station_order))
            )
            future_pred_df = future_pred_df.sort_values(['sort_order', 'Date'])
            future_pred_df = future_pred_df.drop(columns=['sort_order'])

            # Plot and save results
            loading_label.configure(
                text=f"Generating plots and saving results... (Extreme values: {extreme_values_status})")
            self.update()

            # Use our patched function
            summary = patched_plot_and_save_results(df, future_pred_df, target)

            # Restore original plt.show function
            plt.show = original_show

            # Show completion message
            loading_label.configure(
                text=f"Model execution complete!\nResults saved with Station 1 first.\n"
                     f"Extreme values were {extreme_values_status}.\n"
                     "Please click 'Refresh Data' to view updated predictions."
            )

            # Add refresh button
            refresh_button = ctk.CTkButton(
                self.content_frame,
                text="Refresh Data",
                width=150,
                height=30,
                fg_color="#1f6aa5",
                hover_color="#18558a",
                command=self.refresh_data
            )
            refresh_button.pack(pady=20)

        except Exception as e:
            import traceback
            traceback.print_exc()

            # Show error message
            for widget in self.content_frame.winfo_children():
                widget.destroy()

            error_label = ctk.CTkLabel(
                self.content_frame,
                text=f"Error running model:\n{str(e)}",
                font=("Arial", 16),
                text_color="#FF5733"
            )
            error_label.pack(pady=50)

    def refresh_data(self):
        """Reload data after model execution and show only current and future months"""
        try:
            # Reload the data from CSV
            self.data = pd.read_csv("CSV/chlorophyll_predictions_by_station.csv")

            # Convert Date to datetime without filtering initially
            self.data["Date"] = pd.to_datetime(self.data["Date"])
            self.data["Year"] = self.data["Date"].dt.year
            self.data["Month"] = self.data["Date"].dt.month_name()

            # Get current date
            current_date = pd.Timestamp.now()

            # Filter data for current and future dates
            filtered_data = self.data[self.data["Date"].dt.date >= current_date.date()]

            # Find data with chlorophyll values - do this after date filtering
            data_with_chlorophyll = filtered_data.dropna(subset=["Predicted_Chlorophyll"])

            # Get unique years and months from the filtered data with chlorophyll values
            all_years = sorted([str(year) for year in data_with_chlorophyll["Year"].unique()])
            all_months = data_with_chlorophyll["Month"].unique().tolist()

            # Define the month order dictionary
            month_order = {
                "January": 1, "February": 2, "March": 3, "April": 4,
                "May": 5, "June": 6, "July": 7, "August": 8,
                "September": 9, "October": 10, "November": 11, "December": 12
            }

            # Sort months chronologically
            all_months.sort(key=lambda x: month_order.get(x, 13))

            # Create a dictionary of available data by year and month
            self.available_data = {}
            for year in all_years:
                self.available_data[year] = {}
                year_data = data_with_chlorophyll[data_with_chlorophyll["Year"] == int(year)]

                for month in all_months:
                    month_data = year_data[year_data["Month"] == month]
                    self.available_data[year][month] = not month_data.empty

            # Set initial view to current month and year
            initial_year = str(current_date.year)
            initial_month = current_date.strftime("%B")

            # Update year dropdown
            self.year_dropdown.configure(values=all_years)
            if initial_year in all_years:
                self.year_var.set(initial_year)
            elif all_years:
                self.year_var.set(all_years[0])

            # Update month dropdown with available/unavailable indicators
            available_months = []
            unavailable_months = []

            selected_year = self.year_var.get()
            if selected_year in self.available_data:
                for month in all_months:
                    if month in self.available_data[selected_year] and self.available_data[selected_year][month]:
                        available_months.append(month)
                    else:
                        unavailable_months.append(f"⚠️ {month} (No Data)")

            # Combine lists with available months first
            display_months = available_months + unavailable_months

            # Update month dropdown
            self.month_dropdown.configure(values=display_months)

            # Set initial month
            if initial_month in available_months:
                self.month_var.set(initial_month)
            elif available_months:
                self.month_var.set(available_months[0])
            elif unavailable_months:
                self.month_var.set(unavailable_months[0])

            # Configure dropdowns
            self.year_dropdown.configure(command=self.on_year_selected)
            self.month_dropdown.configure(command=self.on_month_selected)

            # Show preview with new data
            self.show_chlorophyll_preview()

        except Exception as e:
            print(f"Error refreshing data: {e}")
            import traceback
            traceback.print_exc()

            for widget in self.content_frame.winfo_children():
                widget.destroy()

            error_label = ctk.CTkLabel(
                self.content_frame,
                text=f"Error refreshing data:\n{str(e)}",
                font=("Segoe UI", 16),
                text_color="#FF5733"
            )
            error_label.pack(pady=50)

    def on_year_selected(self, selected_year):
        """Handle year selection"""
        try:
            # Get all months
            all_months = sorted(list(self.available_data[selected_year].keys()),
                                key=lambda x: {"January": 1, "February": 2, "March": 3, "April": 4,
                                               "May": 5, "June": 6, "July": 7, "August": 8,
                                               "September": 9, "October": 10, "November": 11, "December": 12}.get(x,
                                                                                                                  13))

            # Divide into available and unavailable
            available_months = []
            unavailable_months = []

            for month in all_months:
                if self.available_data[selected_year][month]:
                    available_months.append(month)
                else:
                    unavailable_months.append(f"⚠️ {month} (No Data)")

            # Combine with available months first
            display_months = available_months + unavailable_months

            # Update month dropdown
            self.month_dropdown.configure(values=display_months)

            # Set to first available month if possible
            if available_months:
                self.month_var.set(available_months[0])
            elif unavailable_months:
                self.month_var.set(unavailable_months[0])

            # Update the preview
            self.update_data_selection()
        except Exception as e:
            print(f"Error in on_year_selected: {e}")

    def on_month_selected(self, selected_month):
        """Handle month selection"""
        try:
            # Check if it's an unavailable month
            if selected_month.startswith("⚠️"):
                # Extract the actual month name from the format "⚠️ Month (No Data)"
                month_name = selected_month.split(" ")[1]

                # Get available months
                available_months = []
                for month in self.month_dropdown.cget("values"):
                    if not month.startswith("⚠️"):
                        available_months.append(month)

                # Set to first available month if possible
                if available_months:
                    self.month_var.set(available_months[0])
                    print(f"Switched from unavailable {month_name} to available {available_months[0]}")

                    # Show warning message using standard tkinter messagebox
                    import tkinter.messagebox as tkmb
                    tkmb.showinfo(
                        "No Data Available",
                        f"No data available for {month_name}.\nShowing {available_months[0]} instead."
                    )
                else:
                    # If no available months, just show no data message
                    self.show_no_data_message()
                    return

            # Update the preview
            self.update_data_selection()
        except Exception as e:
            print(f"Error in on_month_selected: {e}")
            self.update_data_selection()  # Still try to update

    # Helper method for showing no data message
    def show_no_data_message(self):
        """Show a message when no data is available for the selected period"""
        # Clear any existing content
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Create a message frame
        message_frame = ctk.CTkFrame(self.content_frame)
        message_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Add message
        ctk.CTkLabel(
            message_frame,
            text="No chlorophyll data available for the selected month and year.",
            font=("Segoe UI", 16)
        ).pack(pady=50)

        # Add suggestion
        ctk.CTkLabel(
            message_frame,
            text="Please select a different month or year.",
            font=("Segoe UI", 14)
        ).pack(pady=10)

    # Helper method for showing error message
    def show_error_message(self, error_text):
        """Show an error message"""
        # Clear any existing content
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Create a message frame
        message_frame = ctk.CTkFrame(self.content_frame)
        message_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Add error message
        ctk.CTkLabel(
            message_frame,
            text="Error displaying chlorophyll data",
            font=("Segoe UI", 16, "bold"),
            text_color="#FF5733"
        ).pack(pady=(50, 10))

        # Add error details
        ctk.CTkLabel(
            message_frame,
            text=error_text,
            font=("Segoe UI", 12),
            wraplength=500
        ).pack(pady=10)

    def show(self):
        """Show the prediction page"""
        self.grid(row=0, column=0, sticky="nsew")
        

