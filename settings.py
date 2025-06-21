import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import re
import sys
import subprocess
import os
import json
import base64
import csv
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
from cryptography.fernet import Fernet
from utils import decrypt_data, generate_key
from login import MASTER_KEY
from audit import get_audit_logger, AUDIT_DIR, AUDIT_FILE
from pathlib import Path


class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, current_user_key, current_user_type="regular", bg_color=None):
        super().__init__(parent, fg_color=bg_color or "#FFFFFF")

        # Show method definition at the top of the class to ensure it's recognized
        self.show = self._show

        # Constants
        self.USER_DATA_FILE = "users.json"
        self.SAVE_DIRECTORY = "user_data_directory"
        self.MASTER_KEY = MASTER_KEY
        self.AUDIT_DIR = AUDIT_DIR
        self.AUDIT_FILE = AUDIT_FILE

        self.ROLE_COLORS = {
            "regular": "#5d7285",  # Secondary text color for regular users
            "superuser": "#1f6aa5",  # Primary blue for super users
            "master": "#FFB74D"  # Orange/Gold for master users
        }

        self.ACTION_DISPLAY_NAMES = {
            "login": "Login",
            "logout": "Logout",
            "failed_login": "Failed Login",
            "password_change": "Password Change",
            "user_created": "User Creation",
            "user_deleted": "User Deletion",
            "user_type_change": "Role Change",
            "data_input": "Data Input",
            "data_export": "Data Export",
            "settings_change": "Settings Change",
            "file_upload": "File Upload",
            "report_generated": "Report Generation",
            "filter_change": "Filter Change",
            "page_access": "Page Access",
            "user_registration": "User Registration",
            "model_error": "Model Error",
            "model_run": "Model Run"
            # Add more mappings as needed
        }

        # Instance variables
        self.parent = parent
        self.current_user_key = current_user_key
        self.current_user_type = current_user_type

        self.primary_color = "#1f6aa5"  # Updated to new blue
        self.primary_light = "#e6f0f7"  # Light blue background
        self.primary_dark = "#17537f"  # Darker blue for hover states
        self.danger_color = "#e74c3c"  # Red for dangerous actions
        self.success_color = "#27ae60"  # Green for success indicators
        self.neutral_color = "#5d7285"  # Updated secondary text color
        self.dark_text = "#2c3e50"  # Updated primary text color
        self.light_text = "#ffffff"  # White for text on dark backgrounds
        self.border_color = "#e1e7ec"  # Updated divider/border color
        self.bg_color = bg_color or "#FFFFFF"  # Match main app bg color
        self.card_bg = "#FFFFFF"  # White for card backgrounds
        self.disabled_bg = "#c4cfd8"  # Disabled button background
        self.disabled_text = "#7d8f9b"  # Disabled button text

        # Font settings
        self.header_font = ctk.CTkFont(family="Segoe UI", size=22, weight="bold")
        self.subheader_font = ctk.CTkFont(family="Segoe UI", size=16, weight="bold")
        self.body_font = ctk.CTkFont(family="Segoe UI", size=13)
        self.small_font = ctk.CTkFont(family="Segoe UI", size=11)
        self.button_font = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        self.body_font_bold = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")

        # Initialize cached dimensions
        self.last_width = 800
        self.last_height = 600

        # Set up basic frame structure
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create placeholder for main container that will be properly set up when shown
        self.main_container = None

        # Flag to track if UI is initialized
        self.is_ui_initialized = False

        # Flag to prevent recursive resizing
        self.is_resizing = False

        # Load user data
        self.user_data = self.load_user_data()
        self.current_user_data = self.user_data.get(current_user_key, {})

    def _show(self):
        self.grid(row=0, column=0, sticky="nsew")

        # Force update of geometry
        self.update_idletasks()

        # Initialize UI if it hasn't been set up yet
        if not self.is_ui_initialized:
            self.setup_ui()
        else:
            # If already initialized, just update dimensions
            self.on_resize()

        # Unbind any previous mousewheel bindings before rebinding to prevent duplicates
        try:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        except:
            pass

        # Rebind the mousewheel events
        self.bind_mousewheel()

        print(f"Settings page shown with dimensions: {self.winfo_width()}x{self.winfo_height()}")

    def load_user_data(self):
        # Load and decrypt user data from JSON file
        try:
            with open(os.path.join(self.SAVE_DIRECTORY, 'users.json'), 'r') as file:
                data = json.load(file)

                # Decrypt the data fields
                decrypted_data = {}
                for user_key, user_info in data.items():
                    # Generate the encryption key
                    salt = user_key.encode()  # Use the username as the salt
                    key = generate_key(MASTER_KEY.decode(), salt)  # Generate the key

                    decrypted_user = {}
                    try:
                        if 'designation' in user_info:
                            decrypted_user['designation'] = decrypt_data(base64.b64decode(user_info['designation']),
                                                                         key)

                        if 'password' in user_info:
                            decrypted_user['password'] = decrypt_data(base64.b64decode(user_info['password']), key)

                        # Decrypt the user type
                        if 'user_type' in user_info:
                            decrypted_user['user_type'] = decrypt_data(base64.b64decode(user_info['user_type']), key)
                        else:
                            decrypted_user['user_type'] = 'regular'  # Default value

                        decrypted_data[user_key] = decrypted_user
                    except Exception as e:
                        print(f"Error decrypting data for user {user_key}: {e}")

                return decrypted_data
        except FileNotFoundError:
            print("User data file not found.")
            return {}
        except json.JSONDecodeError:
            print("Error decoding JSON.")
            return {}

    def encrypt_data(self, data, encryption_key):
        """Encrypt data using Fernet"""
        f = Fernet(encryption_key)
        encrypted_data = f.encrypt(data.encode())
        return encrypted_data

    def save_user_data(self, data):
        """Encrypt and save user data to JSON file"""
        try:
            user_data_encrypted = {}
            for username, user_data in data.items():
                salt = username.encode()
                key = generate_key(self.MASTER_KEY.decode(), salt)

                encrypted_password = self.encrypt_data(user_data['password'], key)
                encrypted_password_b64 = base64.b64encode(encrypted_password).decode()

                encrypted_designation = self.encrypt_data(user_data['designation'], key)
                encrypted_designation_b64 = base64.b64encode(encrypted_designation).decode()

                # Encrypt user type
                user_type = user_data.get('user_type', 'regular')
                encrypted_user_type = self.encrypt_data(user_type, key)
                encrypted_user_type_b64 = base64.b64encode(encrypted_user_type).decode()

                user_data_encrypted[username] = {
                    'password': encrypted_password_b64,
                    'designation': encrypted_designation_b64,
                    'user_type': encrypted_user_type_b64
                }

            json_string = json.dumps(user_data_encrypted)

            filepath = os.path.join(self.SAVE_DIRECTORY, self.USER_DATA_FILE)
            with open(filepath, "w") as file:
                file.write(json_string)
            return True
        except Exception as e:
            messagebox.showerror(title="Error", message=f"Error saving user data: {e}")
            return False

    def setup_ui(self):
        """Create the main UI framework"""
        # Clear existing widgets if any
        if self.main_container:
            self.main_container.destroy()

        # Get current window dimensions
        self.update_idletasks()  # Ensure geometry info is updated

        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        if parent_width < 200:
            parent_width = 800
        if parent_height < 200:
            parent_height = 600

        # Cache these values
        self.last_width = parent_width
        self.last_height = parent_height

        print(f"Setting up UI with dimensions: {parent_width}x{parent_height}")

        self.canvas = ctk.CTkCanvas(self, highlightthickness=0, bg=self._apply_appearance_mode(self.bg_color))
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create a frame inside the canvas to hold all content
        self.main_container = ctk.CTkFrame(self.canvas, fg_color="transparent")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.main_container, anchor="nw")

        # Configure main_container
        self.main_container.grid_columnconfigure(0, weight=1)

        # Create sections
        self.create_header_section()
        self.create_profile_section()
        self.create_settings_section()

        # Add logout section
        self.create_logout_section()

        # Set flag to track that UI is initialized
        self.is_ui_initialized = True

        # Configure events for proper scrolling and resizing
        self.main_container.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # Bind mousewheel events for scrolling
        self.bind_mousewheel()

    def on_frame_configure(self, event):
        """Update the scroll region based on the content frame size"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Adjust the width of the canvas window when canvas is resized"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def bind_mousewheel(self):
        """Bind mousewheel to scroll the canvas"""

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind for Windows
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def on_resize(self, event=None):
        """Handle window resize events with protection against recursion"""
        # Skip if UI not initialized or currently processing a resize
        if not self.is_ui_initialized or not self.main_container or self.is_resizing:
            return

        # Get parent dimensions to avoid self-referencing size issues
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Skip if dimensions are invalid
        if parent_width <= 1 or parent_height <= 1:
            return

        # Only update if size has changed significantly (performance optimization)
        width_diff = abs(parent_width - self.last_width)
        height_diff = abs(parent_height - self.last_height)

        if width_diff < 20 and height_diff < 20:
            return

        # Set resizing flag to prevent recursion
        self.is_resizing = True

        try:
            # Update last known dimensions
            self.last_width = parent_width
            self.last_height = parent_height

            # Update scrollable frame size with maximum width constraint
            self.main_container.configure(
                width=min(max(parent_width - 40, 400), 1200),  # Set both min and max
                height=max(parent_height - 40, 400)
            )
        finally:
            # Reset resizing flag
            self.is_resizing = False

    def create_header_section(self):
        """Create the header with title and welcome message"""
        header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent", corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=30, pady=(20, 0))
        header_frame.grid_columnconfigure(0, weight=1)

        # Welcome message with user's name
        welcome_text = f"Welcome, {self.current_user_key}"
        welcome_label = ctk.CTkLabel(
            header_frame,
            text=welcome_text,
            font=self.header_font,
            text_color=self.dark_text,
            anchor="w"
        )
        welcome_label.pack(anchor="w")

        # Subtitle
        settings_label = ctk.CTkLabel(
            header_frame,
            text="Manage your account settings",
            font=self.body_font,
            text_color=self.neutral_color,
            anchor="w"
        )
        settings_label.pack(anchor="w", pady=(0, 10))

        # Add a subtle divider
        divider = ctk.CTkFrame(header_frame, height=2, fg_color=self.border_color)
        divider.pack(fill="x", pady=(0, 10))

    def create_profile_section(self):
        """Create the profile information card"""
        profile_frame = ctk.CTkFrame(
            self.main_container,
            fg_color=self.card_bg,
            corner_radius=15,
            border_width=1,
            border_color=self.border_color
        )
        profile_frame.grid(row=1, column=0, sticky="ew", padx=30, pady=20)
        profile_frame.grid_columnconfigure(1, weight=1)

        # Section title
        title_label = ctk.CTkLabel(
            profile_frame,
            text="Profile Information",
            font=self.subheader_font,
            text_color=self.dark_text,
            anchor="w"
        )
        title_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(20, 10))

        # Add profile image (left side)
        self.image_container = ctk.CTkFrame(profile_frame, fg_color="transparent", width=160, height=160)
        self.image_container.grid(row=1, column=0, padx=20, pady=20, sticky="n")
        self.image_container.grid_propagate(False)  # Important to maintain size
        self.display_user_image(self.image_container)

        # User information (right side)
        info_frame = ctk.CTkFrame(profile_frame, fg_color="transparent")
        info_frame.grid(row=1, column=1, padx=(0, 20), pady=20, sticky="nsew")

        # Create info fields with a more modern look
        # Username field
        self.add_info_field(info_frame, 0, "Username", self.current_user_key)

        # Designation field
        designation = self.current_user_data.get('designation', 'Not specified')
        self.add_info_field(info_frame, 1, "Designation", designation)

        # Account status (user type)
        user_type = self.current_user_data.get('user_type', 'regular')
        status_color = self.ROLE_COLORS.get(user_type, self.ROLE_COLORS["regular"])
        self.add_info_field(info_frame, 2, "Account Type", user_type.upper(), status_color, use_bold=True)

    def create_settings_section(self):
        """Create a dedicated settings section with improved layout"""
        settings_frame = ctk.CTkFrame(
            self.main_container,
            fg_color=self.card_bg,
            corner_radius=15,
            border_width=1,
            border_color=self.border_color
        )
        settings_frame.grid(row=2, column=0, sticky="ew", padx=30, pady=20)
        settings_frame.grid_columnconfigure(0, weight=1)

        # Section title
        title_label = ctk.CTkLabel(
            settings_frame,
            text="Settings",
            font=self.subheader_font,
            text_color=self.dark_text,
            anchor="w"
        )
        title_label.pack(anchor="w", padx=20, pady=(20, 10))

        # Settings options container
        options_container = ctk.CTkFrame(settings_frame, fg_color="transparent")
        options_container.pack(fill="x", padx=20, pady=(0, 20))

        options_container.grid_columnconfigure(0, weight=1)
        options_container.grid_columnconfigure(1, weight=1)

        # Profile picture setting - "Change" button
        self.create_setting_option(
            options_container, 0, 0,
            "Change Profile Picture",
            "Update your profile image",
            "📷",
            self.upload_image,
            "Change"  # Custom button text
        )

        # Password setting - "Change" button
        self.create_setting_option(
            options_container, 0, 1,
            "Update Password",
            "Change your account password",
            "🔒",
            self.change_password,
            "Update"  # Custom button text
        )

        # Add user management option for master users
        if self.current_user_type == "master":
            # User management - "Manage" button
            self.create_setting_option(
                options_container, 1, 0,
                "Manage Accounts",
                "User management console",
                "👥",
                self.show_user_management,
                "Manage"  # Custom button text
            )

            # Audit trail - "View" button
            self.create_setting_option(
                options_container, 1, 1,
                "Audit Trail",
                "View system activity logs",
                "📋",
                self.show_audit_trail,
                "View"  # Custom button text
            )

        # Add audit trail option for superusers
        elif self.current_user_type == "superuser":
            # Audit trail - "View" button for superusers
            self.create_setting_option(
                options_container, 1, 0,
                "Audit Trail",
                "View system activity logs",
                "📋",
                self.show_audit_trail,
                "View"  # Custom button text
            )

    def create_logout_section(self):
        """Create a simple logout section"""
        logout_frame = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        logout_frame.grid(row=3, column=0, sticky="ew", padx=30, pady=(0, 20))

        # Center the logout button
        logout_frame.grid_columnconfigure(0, weight=1)

        logout_button = ctk.CTkButton(
            logout_frame,
            text="Sign Out",
            font=self.button_font,
            fg_color=self.danger_color,
            hover_color="#c0392b",  # Darker red on hover
            corner_radius=8,
            height=45,
            width=200,
            command=self.logout
        )
        logout_button.grid(row=0, column=0, pady=10)

    def create_setting_option(self, parent, row, col, title, description, icon, command, button_text="Change"):
        """Create a setting option card"""
        option_frame = ctk.CTkFrame(
            parent,
            fg_color=self.primary_light,
            corner_radius=10,
            border_width=0
        )
        option_frame.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)

        # Add minimum height to ensure consistent card sizes
        option_frame.grid_propagate(False)  # Prevent frame from resizing to fit content
        option_frame.configure(height=120)  # Set consistent height for all cards

        # Option content
        content = ctk.CTkFrame(option_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=15)

        # Icon and title in the same row
        header = ctk.CTkFrame(content, fg_color="transparent")
        header.pack(fill="x", pady=(0, 5))

        # Icon
        icon_label = ctk.CTkLabel(
            header,
            text=icon,
            font=ctk.CTkFont(family="Segoe UI", size=24),
            text_color=self.primary_color
        )
        icon_label.pack(side="left", padx=(0, 10))

        # Title
        title_label = ctk.CTkLabel(
            header,
            text=title,
            font=self.body_font,
            text_color=self.dark_text,
            anchor="w"
        )
        title_label.pack(side="left", fill="x", expand=True)

        # Description
        desc_label = ctk.CTkLabel(
            content,
            text=description,
            font=self.small_font,
            text_color=self.neutral_color,
            anchor="w"
        )
        desc_label.pack(anchor="w", pady=(0, 10))

        # Button with custom text
        button = ctk.CTkButton(
            content,
            text=button_text,
            font=self.small_font,
            fg_color=self.primary_color,
            hover_color=self.primary_dark,
            corner_radius=8,
            height=30,
            width=100,
            command=command
        )
        button.pack(anchor="w")

    def add_info_field(self, parent, row, label_text, value_text, value_color=None, use_bold=False):
        """Helper to create a consistent field layout with role color support"""
        # Field container
        field_frame = ctk.CTkFrame(parent, fg_color="transparent")
        field_frame.grid(row=row, column=0, sticky="ew", pady=10)

        # Label with slightly muted color
        label = ctk.CTkLabel(
            field_frame,
            text=label_text,
            font=self.small_font,
            text_color=self.neutral_color,
            anchor="w"
        )
        label.pack(anchor="w")

        # Automatically apply role color if this is a user type field
        if label_text == "Account Type" and not value_color:
            # Convert to lowercase to match dictionary keys
            role = value_text.lower()
            value_color = self.ROLE_COLORS.get(role, self.dark_text)

        # Select font based on use_bold parameter
        font_to_use = self.body_font_bold if use_bold or label_text == "Account Type" else self.body_font

        # Value with prominent color
        value = ctk.CTkLabel(
            field_frame,
            text=value_text,
            font=font_to_use,  # Use bold font if specified
            text_color=value_color or self.dark_text,
            anchor="w"
        )
        value.pack(anchor="w")

    def show_user_management(self):
        """Show the user management in a new window"""
        # Create a new toplevel window
        management_window = ctk.CTkToplevel(self)
        management_window.title("User Management Console")
        management_window.geometry("1000x600")

        management_window.resizable(False, False)

        # Make the window modal
        management_window.grab_set()
        management_window.focus_set()

        # Center the window on the screen
        x = self.winfo_rootx() + (self.winfo_width() // 2) - (1000 // 2)
        y = self.winfo_rooty() + (self.winfo_height() // 2) - (600 // 2)
        management_window.geometry(f"+{x}+{y}")

        # Get the audit logger from the main application
        try:
            from audit import get_audit_logger
            self.audit_logger = get_audit_logger()
        except Exception as e:
            print(f"Could not initialize audit logger: {e}")

        # Create the window content
        header_frame = ctk.CTkFrame(management_window, fg_color=self.primary_color, corner_radius=0, height=60)
        header_frame.pack(fill="x")

        header_label = ctk.CTkLabel(
            header_frame,
            text="User Management Console",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#FFFFFF"
        )
        header_label.place(relx=0.5, rely=0.5, anchor="center")

        # Main content - Light background
        content_frame = ctk.CTkFrame(management_window, fg_color="#F9FAFB")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Main content panel with 70/30 split layout
        main_panel = ctk.CTkFrame(content_frame, fg_color="transparent")
        main_panel.pack(fill="both", expand=True)

        # Configure grid for proportional spacing
        main_panel.grid_columnconfigure(0, weight=7)  # 70% width for table
        main_panel.grid_columnconfigure(1, weight=3)  # 30% width for details
        main_panel.grid_rowconfigure(0, weight=1)

        # Left panel - User accounts with table
        accounts_frame = ctk.CTkFrame(main_panel, fg_color="#FFFFFF", corner_radius=8, border_width=1,
                                      border_color=self.border_color)
        accounts_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 0))

        # Table header and content container
        accounts_container = ctk.CTkFrame(accounts_frame, fg_color="transparent")
        accounts_container.pack(fill="both", expand=True, padx=15, pady=15)

        # Header with title and selection status combined
        header_container = ctk.CTkFrame(accounts_container, fg_color="transparent")
        header_container.pack(fill="x", pady=(0, 5))

        table_header = ctk.CTkLabel(
            header_container,
            text="User Accounts",
            font=self.subheader_font,
            anchor="w"
        )
        table_header.pack(side="left")

        self.selected_header = ctk.CTkLabel(
            header_container,
            text="Select a user to manage",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=self.neutral_color,
            anchor="e"
        )
        self.selected_header.pack(side="right")

        # Filter container below header
        filter_container = ctk.CTkFrame(accounts_container, fg_color="transparent")
        filter_container.pack(fill="x", pady=(5, 15))

        filter_container.grid_columnconfigure(0, weight=0)  # Search label
        filter_container.grid_columnconfigure(1, weight=1)  # Search entry
        filter_container.grid_columnconfigure(2, weight=0)  # Refresh button
        filter_container.grid_columnconfigure(3, weight=0)  # Role label
        filter_container.grid_columnconfigure(4, weight=1)  # Role dropdown

        # Search filter
        search_label = ctk.CTkLabel(filter_container, text="Search:", font=self.body_font)
        search_label.grid(row=0, column=0, sticky="e", padx=(0, 8))

        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", self.filter_users)

        search_entry = ctk.CTkEntry(
            filter_container,
            placeholder_text="Search by username...",
            textvariable=self.search_var,
            width=250,
            height=32,
            corner_radius=6,
            border_width=1,
            border_color=self.border_color
        )
        search_entry.grid(row=0, column=1, sticky="ew", padx=5)

        # Refresh button
        refresh_btn = ctk.CTkButton(
            filter_container,
            text="Refresh",
            font=self.small_font,
            fg_color=self.primary_color,
            hover_color=self.primary_dark,
            corner_radius=6,
            height=32,
            width=90,
            command=self.populate_users_table
        )
        refresh_btn.grid(row=0, column=2, padx=(15, 25))  # Added safe distance

        # User type filter
        type_label = ctk.CTkLabel(filter_container, text="Role:", font=self.body_font)
        type_label.grid(row=0, column=3, sticky="e", padx=(0, 8))

        self.type_filter_var = ctk.StringVar(value="All Types")

        type_filter = ctk.CTkOptionMenu(
            filter_container,
            variable=self.type_filter_var,
            values=["All Types", "Master", "Superuser", "Regular"],
            width=150,
            height=32,
            corner_radius=6,
            fg_color=self.primary_color,
            button_color=self.primary_color,
            button_hover_color=self.primary_dark,
            dropdown_fg_color="#FFFFFF",
            dropdown_hover_color=self.primary_light,
            dropdown_text_color=self.dark_text,
            text_color="#FFFFFF",
            command=lambda x: self.filter_users()
        )
        type_filter.grid(row=0, column=4, sticky="w", padx=5)

        # Create Treeview for users
        columns = ("username", "user_type")
        self.users_table = ttk.Treeview(accounts_container, columns=columns, show="headings", selectmode="browse")

        # Style the treeview
        style = ttk.Style()
        style.theme_use("default")

        # Configure the Treeview style
        style.configure(
            "Treeview",
            background="#FFFFFF",
            foreground="#1F2937",
            rowheight=35,  # Slightly taller rows for better readability
            fieldbackground="#FFFFFF",
            font=("Segoe UI", 12)
        )
        style.configure("Treeview.Heading", font=("Segoe UI", 12, "bold"))

        # Change selected color
        style.map("Treeview", background=[("selected", self.primary_light)],
                  foreground=[("selected", self.primary_dark)])

        # Define column headings
        self.users_table.heading("username", text="Username")
        self.users_table.heading("user_type", text="Account Type")

        # Define column widths
        self.users_table.column("username", width=350, anchor="w")
        self.users_table.column("user_type", width=150, anchor="w")

        # Add vertical scrollbar
        vsb = ttk.Scrollbar(accounts_container, orient="vertical", command=self.users_table.yview)
        self.users_table.configure(yscrollcommand=vsb.set)

        # Pack the scrollbar and treeview
        vsb.pack(side="right", fill="y")
        self.users_table.pack(side="left", fill="both", expand=True)

        # Right panel
        details_frame = ctk.CTkFrame(main_panel, fg_color="#FFFFFF", corner_radius=8, border_width=1,
                                     border_color="#E5E7EB")
        details_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 0))

        # Details container with proper spacing
        details_container = ctk.CTkFrame(details_frame, fg_color="transparent")
        details_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Details header
        details_title = ctk.CTkLabel(
            details_container,
            text="User Details",
            font=self.subheader_font,
            anchor="w"
        )
        details_title.pack(anchor="w", pady=(0, 15))

        # Role indicator for current role
        self.role_indicator = ctk.CTkLabel(
            details_container,
            text="",  # Initially empty
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            anchor="w"
        )
        self.role_indicator.pack(anchor="w", pady=(0, 15))

        # Subtle divider
        divider = ctk.CTkFrame(details_container, height=1, fg_color="#E5E7EB")
        divider.pack(fill="x", pady=(0, 20))

        # Role selection section with minimal styling and smaller footprint
        role_section = ctk.CTkFrame(details_container, fg_color="#F9FAFB", corner_radius=6, border_width=1,
                                    border_color="#E5E7EB")
        role_section.pack(fill="x", pady=(0, 20))

        role_header_frame = ctk.CTkFrame(role_section, fg_color="transparent", height=40)
        role_header_frame.pack(fill="x", padx=15, pady=(12, 0))

        role_header = ctk.CTkLabel(
            role_header_frame,
            text="User Role",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=self.dark_text,
            anchor="w"
        )
        role_header.pack(side="left")

        # Initialize the user type variable
        self.user_type_var = ctk.StringVar(value="regular")

        # Lightweight divider
        divider = ctk.CTkFrame(role_section, height=1, fg_color="#E5E7EB")
        divider.pack(fill="x", padx=15, pady=(12, 0))

        # Type selection container with clean spacing
        self.type_selector = ctk.CTkFrame(role_section, fg_color="transparent")
        self.type_selector.pack(fill="x", padx=15, pady=12)

        # Create compact role options
        self.create_type_option("regular", "Regular User", "Basic access to system features",
                                self.ROLE_COLORS["regular"])
        self.create_type_option("superuser", "Super User", "Advanced access with elevated privileges",
                                self.ROLE_COLORS["superuser"])
        self.create_type_option("master", "Master User", "Full administrative control", self.ROLE_COLORS["master"])

        # Action buttons container at bottom
        apply_container = ctk.CTkFrame(details_container, fg_color="transparent")
        apply_container.pack(fill="x", pady=(5, 0))

        # Apply button
        apply_btn = ctk.CTkButton(
            apply_container,
            text="Apply Role Change",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=self.success_color,
            hover_color="#059669",  # Darker green
            corner_radius=6,
            height=36,  # Smaller height
            width=160,  # Fixed width
            command=self.change_user_type
        )
        apply_btn.pack(side="right")

        # Add a divider after the role change section
        divider2 = ctk.CTkFrame(details_container, height=1, fg_color="#E5E7EB")
        divider2.pack(fill="x", pady=(25, 20))

        # Reset Password section
        reset_header = ctk.CTkLabel(
            details_container,
            text="Reset Password",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=self.dark_text,
            anchor="w"
        )
        reset_header.pack(anchor="w", pady=(0, 10))

        reset_desc = ctk.CTkLabel(
            details_container,
            text="Reset the user's password to the default value.\nThis action cannot be undone.",
            font=self.small_font,
            text_color=self.neutral_color,
            justify="left"
        )
        reset_desc.pack(anchor="w", pady=(0, 15))

        # Reset Password button
        reset_btn = ctk.CTkButton(
            details_container,
            text="Reset Password",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=self.danger_color,
            hover_color="#B91C1C",  # Darker red
            corner_radius=6,
            height=36,
            width=160,
            command=self.reset_user_password
        )
        reset_btn.pack(anchor="w")

        # Status bar at the bottom with minimal styling
        status_frame = ctk.CTkFrame(content_frame, fg_color="transparent", height=30)
        status_frame.pack(fill="x", pady=(15, 0))

        self.user_status_var = ctk.StringVar(value="Ready")
        status_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.user_status_var,
            font=self.small_font,
            text_color=self.neutral_color
        )
        status_label.pack(side="left")

        # Table selection event
        self.users_table.bind('<<TreeviewSelect>>', self.on_user_select)

        # Populate the table
        self.populate_users_table()

    def reset_user_password(self):
        """Reset the selected user's password to the default password"""
        # Check if a user is selected in the users_table
        if not hasattr(self, 'users_table') or not self.users_table.selection():
            self.show_notification("Please select a user first", "warning")
            return

        # Get selected user
        selected_item = self.users_table.selection()[0]
        values = self.users_table.item(selected_item, 'values')

        if not values or len(values) < 2:
            return

        username = values[0]

        if username == self.current_user_key:
            self.show_notification("You cannot reset your own password with this feature", "warning")
            return

        # Default password
        default_password = "Terra#2025"

        # Confirm reset with warning
        confirm = messagebox.askyesno(
            "Confirm Password Reset",
            f"Are you sure you want to reset the password for {username}?\n\n"
            f"The password will be reset to the default value.\n"
            "This action cannot be undone.",
            icon=messagebox.WARNING
        )

        if not confirm:
            return

        # Update user password
        if username in self.user_data:
            # Update the user data with the default password
            self.user_data[username]['password'] = default_password

            # Save the updated user data
            if self.save_user_data(self.user_data):
                # Log the change to audit trail if available
                try:
                    if hasattr(self, 'audit_logger'):
                        self.audit_logger.log_event(
                            self.current_user_key,
                            self.current_user_type,
                            "PASSWORD_CHANGE",
                            f"Reset password for user '{username}' to default"
                        )
                except Exception as e:
                    print(f"Error logging to audit trail: {e}")

                # Update the UI to reflect the change
                self.show_notification(f"Password for {username} has been reset successfully", "success")
            else:
                self.show_notification("Failed to save user data", "error")
        else:
            self.show_notification(f"User {username} not found in the database", "error")

    def show_audit_trail(self):
        """Show the audit trail records from the CSV file"""
        # Create a new toplevel window
        audit_window = ctk.CTkToplevel(self)
        audit_window.title("System Audit Trail")
        audit_window.geometry("1000x600")
        audit_window.resizable(False, False)

        # Make the window modal
        audit_window.grab_set()
        audit_window.focus_set()

        # Center the window on the screen
        x = self.winfo_rootx() + (self.winfo_width() // 2) - (1000 // 2)
        y = self.winfo_rooty() + (self.winfo_height() // 2) - (600 // 2)
        audit_window.geometry(f"+{x}+{y}")

        # Create the window content
        # Header with primary blue background
        header_frame = ctk.CTkFrame(audit_window, fg_color="#1f6aa5", corner_radius=0, height=60)
        header_frame.pack(fill="x")

        header_label = ctk.CTkLabel(
            header_frame,
            text="System Audit Trail",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#FFFFFF"  # White text
        )
        header_label.place(relx=0.5, rely=0.5, anchor="center")

        # Main content with light background
        content_frame = ctk.CTkFrame(audit_window, fg_color="#F9FAFB")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        filter_frame = ctk.CTkFrame(content_frame, fg_color="#FFFFFF", corner_radius=10, border_width=1,
                                    border_color="#e1e7ec", height=80)
        filter_frame.pack(fill="x", pady=(0, 15))
        filter_frame.pack_propagate(False)

        # Filter options
        filter_inner = ctk.CTkFrame(filter_frame, fg_color="transparent")
        filter_inner.pack(fill="both", expand=True, padx=15, pady=10)

        # Configure grid
        filter_inner.grid_columnconfigure(0, weight=0)  # Search label
        filter_inner.grid_columnconfigure(1, weight=0)  # Search entry
        filter_inner.grid_columnconfigure(2, weight=0)  # User label
        filter_inner.grid_columnconfigure(3, weight=0)  # User dropdown
        filter_inner.grid_columnconfigure(4, weight=0)  # Action label
        filter_inner.grid_columnconfigure(5, weight=0)  # Action dropdown
        filter_inner.grid_columnconfigure(6, weight=1)  # Empty space to push buttons right
        filter_inner.grid_columnconfigure(7, weight=0)  # Refresh button
        filter_inner.grid_columnconfigure(8, weight=0)  # Export button

        # Search filter
        search_label = ctk.CTkLabel(filter_inner, text="Search:", font=self.body_font, text_color="#2c3e50")
        search_label.grid(row=0, column=0, sticky="e", padx=(10, 5), pady=10)

        self.audit_search_var = ctk.StringVar()
        self.audit_search_var.trace_add("write", lambda *args: self.filter_audit_logs())

        # Search entry
        search_entry = ctk.CTkEntry(
            filter_inner,
            placeholder_text="Search in logs...",
            textvariable=self.audit_search_var,
            width=150,  # Original width
            height=32,
            corner_radius=8,
            border_width=1,
            border_color="#c4cfd8",
            placeholder_text_color="#5d7285"  # Secondary text color
        )
        search_entry.grid(row=0, column=1, sticky="w", padx=5, pady=10)

        # User filter
        user_label = ctk.CTkLabel(filter_inner, text="User:", font=self.body_font, text_color="#2c3e50")
        user_label.grid(row=0, column=2, sticky="e", padx=(10, 5), pady=10)

        self.user_filter_var = ctk.StringVar(value="All Users")

        # User filter dropdown
        user_filter = ctk.CTkOptionMenu(
            filter_inner,
            variable=self.user_filter_var,
            values=["All Users"],  
            width=150,  # Original width
            height=32,
            corner_radius=8,
            fg_color="#1f6aa5",  # Primary blue
            button_color="#1f6aa5",  # Primary blue
            button_hover_color="#17537f",  # Darker blue for hover
            dropdown_fg_color="#FFFFFF",  # White background for dropdown
            dropdown_hover_color="#e6f0f7",  # Light blue hover
            dropdown_text_color="#2c3e50",  # Primary text color for dropdown
            text_color="#FFFFFF",  # White text for button
            command=lambda x: self.filter_audit_logs()
        )
        user_filter.grid(row=0, column=3, sticky="w", padx=5, pady=10)
        self.user_filter = user_filter

        # Action filter
        action_label = ctk.CTkLabel(filter_inner, text="Action:", font=self.body_font, text_color="#2c3e50")
        action_label.grid(row=0, column=4, sticky="e", padx=(10, 5), pady=10)

        self.action_filter_var = ctk.StringVar(value="All Actions")

        # Action filter dropdown
        action_filter = ctk.CTkOptionMenu(
            filter_inner,
            variable=self.action_filter_var,
            values=["All Actions"],
            width=150,  # Original width
            height=32,
            corner_radius=8,
            fg_color="#1f6aa5",  # Primary blue
            button_color="#1f6aa5",  # Primary blue
            button_hover_color="#17537f",  # Darker blue for hover
            dropdown_fg_color="#FFFFFF",  # White background for dropdown
            dropdown_hover_color="#e6f0f7",  # Light blue hover
            dropdown_text_color="#2c3e50",  # Primary text color for dropdown
            text_color="#FFFFFF",  # White text for button
            command=lambda x: self.filter_audit_logs()
        )
        action_filter.grid(row=0, column=5, sticky="w", padx=5, pady=10)
        self.action_filter = action_filter

        # Refresh button
        refresh_btn = ctk.CTkButton(
            filter_inner,
            text="Refresh",
            font=self.small_font,
            fg_color="#1f6aa5",  # Primary blue
            hover_color="#17537f",  # Darker blue for hover
            text_color="#FFFFFF",  # White text
            corner_radius=8,
            height=32,
            width=100,  # Original width
            command=self.load_audit_logs
        )
        refresh_btn.grid(row=0, column=7, padx=5, pady=10)

        # Export button
        export_btn = ctk.CTkButton(
            filter_inner,
            text="Export",
            font=self.small_font,
            fg_color="#27ae60",  # Success green
            hover_color="#059669",  # Darker green for hover
            text_color="#FFFFFF",  # White text
            corner_radius=8,
            height=32,
            width=100,  # Original width
            command=self.export_audit_logs
        )
        export_btn.grid(row=0, column=8, padx=5, pady=10)

        # Main logs display area with Treeview (table)
        log_frame = ctk.CTkFrame(content_frame, fg_color="#FFFFFF", corner_radius=10, border_width=1,
                                 border_color="#e1e7ec")
        log_frame.pack(fill="both", expand=True)

        tree_container = ctk.CTkFrame(log_frame, fg_color="transparent")
        tree_container.pack(fill="both", expand=True, padx=15, pady=15)

        # Create Treeview with word wrap for the details column
        columns = ("timestamp", "username", "user_type", "action", "details")
        self.audit_tree = ttk.Treeview(tree_container, columns=columns, show="headings")

        # Style the treeview
        style = ttk.Style()
        style.theme_use("default")

        # Configure the Treeview style with a variable row height to accommodate wrapped text
        style.configure(
            "Treeview",
            background="#FFFFFF",
            foreground="#2c3e50",  # Primary text color
            rowheight=50, 
            fieldbackground="#FFFFFF",
            font=("Segoe UI", 11)
        )
        style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"), foreground="#2c3e50")  # Primary text color
        style.map("Treeview",
                  background=[("selected", "#e6f0f7")],  # Light blue for selection
                  foreground=[("selected", "#1f6aa5")])  # Primary blue for selected text

        # Define column headings
        self.audit_tree.heading("timestamp", text="Timestamp")
        self.audit_tree.heading("username", text="Username")
        self.audit_tree.heading("user_type", text="User Type")
        self.audit_tree.heading("action", text="Action")
        self.audit_tree.heading("details", text="Details")

        # Define column widths
        self.audit_tree.column("timestamp", width=180, anchor="w", minwidth=180)
        self.audit_tree.column("username", width=150, anchor="w", minwidth=150)
        self.audit_tree.column("user_type", width=120, anchor="w", minwidth=120)
        self.audit_tree.column("action", width=150, anchor="w", minwidth=150)
        self.audit_tree.column("details", width=350, anchor="w", minwidth=350)  # Wider details column

        # Create vertical scrollbar
        vsb = ttk.Scrollbar(tree_container, orient="vertical", command=self.audit_tree.yview)

        # Configure the Treeview to use the scrollbar
        self.audit_tree.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        self.audit_tree.pack(side="left", fill="both", expand=True)

        # Status bar
        status_frame = ctk.CTkFrame(content_frame, fg_color="transparent", height=30)
        status_frame.pack(fill="x", pady=(15, 0))

        self.status_var = ctk.StringVar(value="Loading audit logs...")
        status_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.status_var,
            font=self.small_font,
            text_color="#5d7285"  # Secondary text color
        )
        status_label.pack(side="left")

        # Bind a custom function to handle row display after inserting items
        self.audit_tree.bind("<<TreeviewOpen>>", self.adjust_row_heights)

        # Load the audit logs
        self.load_audit_logs()

    def adjust_row_heights(self, event=None):
        """Adjust row heights based on the content of details column"""
        for item_id in self.audit_tree.get_children():
            # Get the text in the details column
            values = self.audit_tree.item(item_id, "values")
            if len(values) >= 5:  # Ensure the details column exists
                details_text = values[4]
                approx_chars_per_line = 60
                num_lines = max(1, (len(details_text) // approx_chars_per_line) + 1)

                if num_lines > 1:
                    # Height per line
                    row_height = (num_lines * 20) + 10
                    # Apply the custom height
                    self.audit_tree.item(item_id, tags=(f"multiline_{row_height}",))

                    # Configure the tag with the specific height
                    self.audit_tree.tag_configure(f"multiline_{row_height}", font=("Segoe UI", 11))

    def load_audit_logs(self):
        """Load audit logs from the CSV file and populate the treeview"""
        # Clear existing data
        for item in self.audit_tree.get_children():
            self.audit_tree.delete(item)

        # Update status
        self.status_var.set("Loading audit logs...")

        # Path to the audit log file
        audit_file_path = Path(self.AUDIT_DIR) / self.AUDIT_FILE

        # Check if the file exists
        if not audit_file_path.exists():
            self.status_var.set("No audit logs found.")
            return

        try:
            # Read all logs from the CSV file
            all_logs = []
            unique_users = set()
            unique_actions = set()

            with open(audit_file_path, 'r', newline='') as file:
                reader = csv.reader(file)
                headers = next(reader)  # Skip header row

                for row in reader:
                    # Make sure row has all required data
                    if len(row) >= 5:
                        timestamp, username, user_type, action, details = row[0], row[1], row[2], row[3], row[4]

                        # Add to unique sets for filters
                        unique_users.add(username)
                        unique_actions.add(action)

                        # Convert action code to proper noun display format
                        display_action = self.ACTION_DISPLAY_NAMES.get(action.lower(), action)

                        # Format the details text to have word-wrapping with newlines
                        formatted_details = self.format_details_for_wrapping(details)

                        # Add to all logs list with the proper noun display format
                        all_logs.append((timestamp, username, user_type, display_action, formatted_details))

            # Sort logs by timestamp
            all_logs.sort(reverse=True)

            # Populate the treeview
            for log in all_logs:
                self.audit_tree.insert("", "end", values=log)

            # Update the filters
            self.update_filter_options(sorted(unique_users), sorted(unique_actions))

            # Update status
            self.status_var.set(f"Loaded {len(all_logs)} audit log entries.")

            # After all items are inserted, adjust row heights
            self.audit_tree.after(100, self.adjust_row_heights)

        except Exception as e:
            self.status_var.set(f"Error loading audit logs: {str(e)}")
            messagebox.showerror("Error", f"Failed to load audit logs: {str(e)}")

    def update_filter_options(self, users, actions):
        """Update the filter dropdown options with display names"""
        # Update user filter
        user_values = ["All Users"] + list(users)
        self.user_filter.configure(values=user_values)

        # Create action display values with proper noun formats
        action_display_values = ["All Actions"]

        # Create mapping dictionaries
        self.code_to_display_map = {}  # Maps action codes to display names
        self.display_to_code_map = {}  # Maps display names back to codes

        for action_code in actions:
            # Get the display version of the action code
            display_name = self.ACTION_DISPLAY_NAMES.get(action_code.lower(), action_code)

            # Add to the values list for the dropdown
            action_display_values.append(display_name)

            # Store the mappings
            self.code_to_display_map[action_code] = display_name
            self.display_to_code_map[display_name] = action_code

        # Update action filter with display names
        self.action_filter.configure(values=action_display_values)

    def format_details_for_wrapping(self, details):
        """Format the details text to have word-wrapping with newlines"""
        if len(details) > 50:
            words = details.split()
            formatted_text = ""
            line_length = 0

            for word in words:
                if line_length + len(word) + 1 > 50:
                    formatted_text += "\n" + word + " "
                    line_length = len(word) + 1
                else:
                    formatted_text += word + " "
                    line_length += len(word) + 1

            return formatted_text.strip()
        return details

    def filter_audit_logs(self):
        """Filter the audit logs based on the current filter settings"""
        search_text = self.audit_search_var.get().lower()
        user_filter = self.user_filter_var.get()
        action_filter_display = self.action_filter_var.get()

        # Convert display action filter back to code for filtering
        if action_filter_display != "All Actions" and hasattr(self, 'display_to_code_map'):
            action_filter = self.display_to_code_map.get(action_filter_display, action_filter_display)
        else:
            action_filter = "All Actions"

        # Clear existing data
        for item in self.audit_tree.get_children():
            self.audit_tree.delete(item)

        # Path to the audit log file
        audit_file_path = Path(self.AUDIT_DIR) / self.AUDIT_FILE

        if not audit_file_path.exists():
            return

        try:
            # Read and filter logs
            filtered_logs = []

            with open(audit_file_path, 'r', newline='') as file:
                reader = csv.reader(file)
                headers = next(reader)  # Skip header row

                for row in reader:
                    # Make sure row has all required data
                    if len(row) >= 5:
                        timestamp, username, user_type, action, details = row[0], row[1], row[2], row[3], row[4]

                        # Apply user filter
                        if user_filter != "All Users" and username != user_filter:
                            continue

                        # Apply action filter (using original action code)
                        if action_filter != "All Actions" and action != action_filter:
                            continue

                        # Convert action code to proper noun display format
                        display_action = self.ACTION_DISPLAY_NAMES.get(action.lower(), action)

                        # Apply search text filter across all fields
                        if search_text and not (
                                search_text in timestamp.lower() or
                                search_text in username.lower() or
                                search_text in user_type.lower() or
                                search_text in action.lower() or
                                search_text in display_action.lower() or
                                search_text in details.lower()
                        ):
                            continue

                        # Format the details text for wrapping
                        formatted_details = self.format_details_for_wrapping(details)

                        # Add to filtered logs with display action
                        filtered_logs.append((timestamp, username, user_type, display_action, formatted_details))

            # Sort logs by timestamp (newest first)
            filtered_logs.sort(reverse=True)

            # Populate the treeview with filtered data
            for log in filtered_logs:
                self.audit_tree.insert("", "end", values=log)

            # Update status
            self.status_var.set(f"Showing {len(filtered_logs)} audit log entries.")

            # After all items are inserted, adjust row heights
            self.audit_tree.after(100, self.adjust_row_heights)

        except Exception as e:
            self.status_var.set(f"Error filtering audit logs: {str(e)}")

    def export_audit_logs(self):
        """Export the filtered audit logs to a CSV file"""
        # Get the current filtered data (with display names)
        filtered_data = []
        for item_id in self.audit_tree.get_children():
            values = self.audit_tree.item(item_id, "values")
            filtered_data.append(values)

        if not filtered_data:
            messagebox.showinfo("Export", "No data to export.")
            return

        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Export Audit Logs"
        )

        if not file_path:
            return

        try:
            # Write the data to the file
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write headers
                writer.writerow(["Timestamp", "Username", "User Type", "Action", "Details"])
                # Write data (already contains display action names)
                writer.writerows(filtered_data)

            messagebox.showinfo("Export Successful", f"Audit logs exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Export Failed", f"Error exporting audit logs: {str(e)}")

    # Redirects for old method names
    def open_user_management(self):
        """Redirects to the new implementation"""
        self.show_user_management()

    def open_audit_trail(self):
        """Redirects to the new implementation"""
        self.show_audit_trail()

    def create_type_option(self, type_value, title, description, color):
        """Create a more compact radio option"""
        option_frame = ctk.CTkFrame(self.type_selector, fg_color="transparent")
        option_frame.pack(fill="x", pady=4)

        # Radio button
        radio = ctk.CTkRadioButton(
            option_frame,
            text="",
            variable=self.user_type_var,
            value=type_value,
            fg_color=color,
            border_color="#E5E7EB",
            width=16, 
            height=16 
        )
        radio.pack(side="left", padx=(0, 8)) 

        # Create text container
        text_frame = ctk.CTkFrame(option_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="x", expand=True)

        # Title and description
        title_container = ctk.CTkFrame(text_frame, fg_color="transparent")
        title_container.pack(fill="x")

        # Title 
        title_label = ctk.CTkLabel(
            title_container,
            text=title,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=self.dark_text,
            anchor="w"
        )
        title_label.pack(side="left")

        # Description
        desc_label = ctk.CTkLabel(
            title_container,
            text=f" — {description}",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=self.neutral_color,
            anchor="w"
        )
        desc_label.pack(side="left")

        # Small colored indicator on the right
        indicator = ctk.CTkFrame(
            option_frame,
            width=6,
            height=16,
            fg_color=color,
            corner_radius=2
        )
        indicator.pack(side="right", padx=(2, 0))

    def filter_users(self, *args):
        """Filter users table based on search text and role filter"""
        search_text = self.search_var.get().lower()
        type_filter = self.type_filter_var.get() if hasattr(self, 'type_filter_var') else "All Types"

        # Clear existing data
        if hasattr(self, 'users_table'):
            for item in self.users_table.get_children():
                self.users_table.delete(item)
        else:
            return

        sorted_users = sorted(self.user_data.items(), key=lambda x: x[0].lower())
        count = 0

        for username, user_data in sorted_users:
            # Apply text filter
            if search_text and search_text not in username.lower():
                continue

            # Apply type filter
            user_type = user_data.get('user_type', 'regular')
            if type_filter != "All Types" and type_filter.lower() != user_type:
                continue

            # Insert into table
            item_id = self.users_table.insert("", "end", values=(username, user_type.capitalize()))

            # Apply color tag
            if user_type == "master":
                self.users_table.tag_configure("master", foreground=self.ROLE_COLORS["master"])
                self.users_table.item(item_id, tags=("master",))
            elif user_type == "superuser":
                self.users_table.tag_configure("superuser", foreground=self.ROLE_COLORS["superuser"])
                self.users_table.item(item_id, tags=("superuser",))
            else:
                self.users_table.tag_configure("regular", foreground=self.ROLE_COLORS["regular"])
                self.users_table.item(item_id, tags=("regular",))

            count += 1

        # Update status with count information
        if hasattr(self, 'user_status_var'):
            if count == 0:
                self.user_status_var.set(f"No users match your filters")
            else:
                self.user_status_var.set(f"Showing {count} user{'s' if count != 1 else ''}")

        # Reset selection header if no users are found
        if count == 0 and hasattr(self, 'selected_header'):
            self.selected_header.configure(text="Select a user to manage")

        # Also clear role indicator if no users found
        if count == 0 and hasattr(self, 'role_indicator'):
            self.role_indicator.configure(text="")

    def populate_users_table(self, filter_text=None):
        """Populate the table with users and their types"""
        # Clear existing data
        for item in self.users_table.get_children():
            self.users_table.delete(item)

        sorted_users = sorted(self.user_data.items(), key=lambda x: x[0].lower())
        count = 0

        for username, user_data in sorted_users:
            if filter_text and filter_text.lower() not in username.lower():
                continue

            user_type = user_data.get('user_type', 'regular')

            # Insert into table
            item_id = self.users_table.insert("", "end", values=(username, user_type.capitalize()))
            count += 1

            # Apply color tag based on user type
            if user_type == "master":
                self.users_table.tag_configure("master", foreground=self.ROLE_COLORS["master"])
                self.users_table.item(item_id, tags=("master",))
            elif user_type == "superuser":
                self.users_table.tag_configure("superuser", foreground=self.ROLE_COLORS["superuser"])
                self.users_table.item(item_id, tags=("superuser",))
            else:
                self.users_table.tag_configure("regular", foreground=self.ROLE_COLORS["regular"])
                self.users_table.item(item_id, tags=("regular",))

        # Update status with count information
        if count == 0:
            self.user_status_var.set(f"No users match your filters")
        else:
            self.user_status_var.set(f"Showing {count} user{'s' if count != 1 else ''}")

    def on_user_select(self, event):
        """Handle user selection from treeview table"""
        if not hasattr(self, 'users_table') or not self.users_table.selection():
            if hasattr(self, 'selected_header'):
                self.selected_header.configure(text="Select a user to manage")

            if hasattr(self, 'role_indicator'):
                self.role_indicator.configure(text="")
            return

        # Get selected item (username and type)
        selected_item = self.users_table.selection()[0]
        values = self.users_table.item(selected_item, 'values')

        if not values or len(values) < 2:
            return

        username = values[0]
        current_type = values[1].lower()  # Convert to lowercase for consistency

        # Get role color
        role_color = self.ROLE_COLORS.get(current_type, self.dark_text)

        # Update the selected user header
        if hasattr(self, 'selected_header'):
            self.selected_header.configure(
                text=f"Managing: {username}",
                text_color=self.primary_color  # Use primary color for visibility
            )

        # Update role indicator
        if hasattr(self, 'role_indicator'):
            self.role_indicator.configure(
                text=f"Current Role: {current_type.upper()}",
                text_color=role_color
            )

        # Set the radio button to match the current user type
        if hasattr(self, 'user_type_var'):
            self.user_type_var.set(current_type)

    def change_user_type(self):
        """Change the selected user's type with confirmation and system-wide update"""
        # Check if a user is selected in the users_table
        if not hasattr(self, 'users_table') or not self.users_table.selection():
            self.show_notification("Please select a user first", "warning")
            return

        # Get selected user
        selected_item = self.users_table.selection()[0]
        values = self.users_table.item(selected_item, 'values')

        if not values or len(values) < 2:
            return

        username = values[0]
        current_type = values[1].lower()  # Convert to lowercase for consistency

        # Don't allow changing your own user type
        if username == self.current_user_key:
            self.show_notification("You cannot change your own user type", "warning")
            return

        # Get new user type from the radio buttons
        new_user_type = self.user_type_var.get()

        if current_type == new_user_type:
            self.show_notification(f"No change: {username} is already a {current_type} user", "info")
            return
        # Confirm change with more detailed information
        confirm = messagebox.askyesno(
            "Confirm Role Change",
            f"Are you sure you want to change {username}'s role?\n\n"
            f"Current role: {current_type.upper()}\n"
            f"New role: {new_user_type.upper()}\n\n"
            "This change will apply throughout the entire system.",
            icon=messagebox.WARNING
        )

        if not confirm:
            return

        # Update user type
        if username in self.user_data:
            # Store the old type for auditing
            old_type = self.user_data[username].get('user_type', 'regular')

            # Update the user data
            self.user_data[username]['user_type'] = new_user_type

            # Save the updated user data
            if self.save_user_data(self.user_data):
                # Log the change to audit trail if available
                try:
                    if hasattr(self, 'audit_logger'):
                        self.audit_logger.log_event(
                            self.current_user_key,
                            self.current_user_type,
                            "USER_TYPE_CHANGE",
                            f"Changed user '{username}' from {old_type} to {new_user_type}"
                        )
                except Exception as e:
                    print(f"Error logging to audit trail: {e}")

                # Update the UI to reflect the change
                self.show_notification(f"User {username} updated to {new_user_type} successfully", "success")

                # Update the selection in the table
                self.users_table.item(selected_item, values=(username, new_user_type.capitalize()))

                # Apply color tag based on new user type
                if new_user_type == "master":
                    self.users_table.tag_configure("master", foreground=self.ROLE_COLORS["master"])
                    self.users_table.item(selected_item, tags=("master",))
                elif new_user_type == "superuser":
                    self.users_table.tag_configure("superuser", foreground=self.ROLE_COLORS["superuser"])
                    self.users_table.item(selected_item, tags=("superuser",))
                else:
                    self.users_table.tag_configure("regular", foreground=self.ROLE_COLORS["regular"])
                    self.users_table.item(selected_item, tags=("regular",))

                # Update the role indicator
                if hasattr(self, 'role_indicator'):
                    role_color = self.ROLE_COLORS.get(new_user_type, self.dark_text)
                    self.role_indicator.configure(
                        text=f"Current Role: {new_user_type.upper()}",
                        text_color=role_color
                    )

                self.filter_users()
            else:
                self.show_notification("Failed to save user data", "error")
        else:
            self.show_notification(f"User {username} not found in the database", "error")

    def show_notification(self, message, type="info"):
        """Show a notification"""
        icon = "ℹ️" if type == "info" else "✅" if type == "success" else "⚠️" if type == "warning" else "❌"

        if type == "error":
            messagebox.showerror("Error", f"{icon} {message}")
        elif type == "warning":
            messagebox.showwarning("Warning", f"{icon} {message}")
        elif type == "success":
            messagebox.showinfo("Success", f"{icon} {message}")
        else:
            messagebox.showinfo("Information", f"{icon} {message}")

    def create_initials_placeholder(self, container=None):
        """Create a circular placeholder with initials if image fails"""
        # If no container is provided, create a new one
        if container is None:
            container = ctk.CTkFrame(self, fg_color="transparent")
            container.pack(padx=10, pady=10)

        # Clear any existing widgets in the container
        for widget in container.winfo_children():
            widget.destroy()

        # Create container frame
        container_frame = ctk.CTkFrame(
            container,
            fg_color="transparent",
            width=140,
            height=140
        )
        container_frame.pack(padx=5, pady=5)
        container_frame.grid_propagate(False)

        # Create a perfect circle with the user's initials
        initial_bg = ctk.CTkFrame(
            container_frame,
            fg_color=self.primary_color,
            corner_radius=70,  # Half of width/height for perfect circle
            width=140,
            height=140
        )
        initial_bg.place(relx=0.5, rely=0.5, anchor="center")

        # Get the first letter of the username
        initial = self.current_user_key[0].upper() if self.current_user_key else "U"

        # Display the initial
        initial_label = ctk.CTkLabel(
            initial_bg,
            text=initial,
            font=ctk.CTkFont(family="Segoe UI", size=48, weight="bold"),
            text_color="#FFFFFF"
        )
        initial_label.place(relx=0.5, rely=0.5, anchor="center")

    def display_user_image(self, container=None):
        """Display the user profile image with a circular frame"""
        # If no container is provided, create a new one
        if container is None:
            container = ctk.CTkFrame(self, fg_color="transparent")
            container.pack(padx=10, pady=10)

        # Clear any existing widgets in the container
        for widget in container.winfo_children():
            widget.destroy()

        # Construct the user-specific image path
        user_image_path = os.path.join(self.SAVE_DIRECTORY, f"{self.current_user_key}_profile_image.png")
        default_image_path = os.path.join(self.SAVE_DIRECTORY, 'default_profile_image.jpg')

        # Check if the user-specific image exists
        if os.path.exists(user_image_path):
            image_path = user_image_path
        elif os.path.exists(default_image_path):
            image_path = default_image_path
        else:
            print("No profile image found for user and no default image available.")
            self.create_initials_placeholder(container)
            return

        try:
            # Load and process the image
            img = Image.open(image_path)
            # Create a square crop from the center of the image
            width, height = img.size
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            right = left + size
            bottom = top + size
            img = img.crop((left, top, right, bottom))
            img = img.resize((140, 140), Image.LANCZOS)

            # Create a circular mask
            mask = Image.new('L', (140, 140), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, 140, 140), fill=255)

            # Apply the mask to create a circular image
            img_rgba = img.convert("RGBA")
            circle_img = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
            circle_img.paste(img_rgba, (0, 0), mask)

            # Create a CTkImage from the circular image
            self.user_image = ctk.CTkImage(
                light_image=circle_img,
                dark_image=circle_img,
                size=(140, 140)
            )

            # Create a circular frame with a border
            frame = ctk.CTkFrame(
                container,
                fg_color="white",
                corner_radius=70, 
                border_width=2,
                border_color="#E5E7EB",
                width=144, 
                height=144
            )
            frame.pack(padx=5, pady=5)
            frame.grid_propagate(False) 

            # Display the image
            image_label = ctk.CTkLabel(
                frame,
                image=self.user_image,
                text=""
            )
            image_label.place(relx=0.5, rely=0.5, anchor="center")

        except Exception as e:
            print(f"Error displaying user image: {e}")
            # Create a placeholder with initials if image fails
            self.create_initials_placeholder(container)

    def upload_image(self):
        """Allow user to upload a profile picture"""
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(
            title="Select Profile Picture",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")],
            initialdir="/"
        )

        if not file_path:
            return

        try:
            # Create target directory if it doesn't exist
            if not os.path.exists(self.SAVE_DIRECTORY):
                os.makedirs(self.SAVE_DIRECTORY)

            # Save the selected image with the user's key as the filename
            new_image_path = os.path.join(self.SAVE_DIRECTORY, f"{self.current_user_key}_profile_image.png")

            # Process the image
            img = Image.open(file_path)
            img = img.resize((300, 300), Image.LANCZOS)  # Resize image to fit
            img.save(new_image_path)  # Save the new image

            # Refresh the displayed image
            self.refresh_profile_image()
            self.show_notification("Profile picture updated successfully", "success")
        except Exception as e:
            self.show_notification(f"Error uploading image: {str(e)}", "error")

    def refresh_profile_image(self):
        """Refresh the profile image after an update"""
        # Clear the current image container
        for widget in self.image_container.winfo_children():
            widget.destroy()

        # Display the updated image
        self.display_user_image(self.image_container)

    def change_password(self):
        """Handle password change flow with validation"""

        def is_valid_password(password):
            """Check if password meets requirements"""
            pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+.,])[A-Za-z\d!@#$%^&*()_+.,]{8,}$"
            return bool(re.match(pattern, password))

        # Create a custom password dialog
        password_dialog = ctk.CTkToplevel(self)
        password_dialog.title("Change Password")
        password_dialog.geometry("400x420") 
        password_dialog.resizable(False, False)
        password_dialog.grab_set()  # Make dialog modal

        # Center the dialog on the parent window
        password_dialog.geometry(f"+{self.parent.winfo_rootx() + 50}+{self.parent.winfo_rooty() + 50}")

        # Dialog header
        header_frame = ctk.CTkFrame(password_dialog, fg_color=self.primary_color, corner_radius=0)
        header_frame.pack(fill="x", pady=(0, 20))

        header_label = ctk.CTkLabel(
            header_frame,
            text="Change Your Password",
            font=self.subheader_font,
            text_color=self.light_text
        )
        header_label.pack(pady=15)

        # Create content frame with padding
        content_frame = ctk.CTkFrame(password_dialog, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Current password field
        current_label = ctk.CTkLabel(
            content_frame,
            text="Current Password",
            font=self.body_font,
            anchor="w"
        )
        current_label.pack(anchor="w", pady=(0, 5))

        current_entry = ctk.CTkEntry(
            content_frame,
            placeholder_text="Enter current password",
            font=self.body_font,
            show="•",
            height=40,
            corner_radius=8
        )
        current_entry.pack(fill="x", pady=(0, 15))

        # New password field
        new_label = ctk.CTkLabel(
            content_frame,
            text="New Password",
            font=self.body_font,
            anchor="w"
        )
        new_label.pack(anchor="w", pady=(0, 5))

        new_entry = ctk.CTkEntry(
            content_frame,
            placeholder_text="Enter new password",
            font=self.body_font,
            show="•",
            height=40,
            corner_radius=8
        )
        new_entry.pack(fill="x", pady=(0, 5))

        # Password requirements note
        requirements = ctk.CTkLabel(
            content_frame,
            text="Password must contain at least 8 characters, including\nupper/lowercase letters, numbers and symbols",
            font=self.small_font,
            text_color=self.neutral_color,
            justify="left"
        )
        requirements.pack(anchor="w", pady=(0, 15))

        # Confirm password field
        confirm_label = ctk.CTkLabel(
            content_frame,
            text="Confirm New Password",
            font=self.body_font,
            anchor="w"
        )
        confirm_label.pack(anchor="w", pady=(0, 5))

        confirm_entry = ctk.CTkEntry(
            content_frame,
            placeholder_text="Confirm new password",
            font=self.body_font,
            show="•",
            height=40,
            corner_radius=8
        )
        confirm_entry.pack(fill="x", pady=(0, 15)) 

        # Function to handle password change
        def do_change_password():
            # Get values from entries
            current_password = current_entry.get()
            new_password = new_entry.get()
            confirm_password = confirm_entry.get()

            # Validate inputs
            if not current_password:
                error_var.set("Please enter your current password")
                return

            if not new_password:
                error_var.set("Please enter a new password")
                return

            if not is_valid_password(new_password):
                error_var.set("Password doesn't meet the requirements")
                return

            if new_password != confirm_password:
                error_var.set("New passwords don't match")
                return

            # Check current password
            stored_password = self.current_user_data.get('password', '')
            if current_password != stored_password:
                error_var.set("Current password is incorrect")
                return

            # Update the password
            try:
                self.current_user_data['password'] = new_password
                self.user_data[self.current_user_key] = self.current_user_data

                if self.save_user_data(self.user_data):
                    password_dialog.destroy()
                    self.show_notification("Password updated successfully", "success")
                else:
                    error_var.set("Failed to save new password")
            except Exception as e:
                error_var.set(f"Error: {str(e)}")

        # Change password button
        change_btn = ctk.CTkButton(
            content_frame,
            text="Change Password",
            font=self.body_font,
            fg_color=self.success_color,
            hover_color="#059669",
            corner_radius=8,
            height=40,
            command=do_change_password
        )
        change_btn.pack(fill="x", pady=(0, 10))

        # Cancel button
        cancel_btn = ctk.CTkButton(
            content_frame,
            text="Cancel",
            font=self.body_font,
            fg_color=self.neutral_color,
            hover_color="#374151",
            corner_radius=8,
            height=40,
            command=password_dialog.destroy
        )
        cancel_btn.pack(fill="x")

        # Error message label
        error_var = ctk.StringVar(value="")
        error_label = ctk.CTkLabel(
            content_frame,
            textvariable=error_var,
            font=self.small_font,
            text_color=self.danger_color
        )
        error_label.pack(fill="x", pady=(10, 0))

    def logout(self):
        """Handle user logout with confirmation"""
        # Create a custom confirmation dialog
        confirm_dialog = ctk.CTkToplevel(self)
        confirm_dialog.title("Confirm Logout")
        confirm_dialog.geometry("350x200")
        confirm_dialog.resizable(False, False)
        confirm_dialog.grab_set()  # Make dialog modal

        # Center the dialog on the parent window
        confirm_dialog.geometry(f"+{self.parent.winfo_rootx() + 50}+{self.parent.winfo_rooty() + 50}")

        # Dialog content
        content_frame = ctk.CTkFrame(confirm_dialog, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Icon
        icon_label = ctk.CTkLabel(
            content_frame,
            text="🚪",
            font=ctk.CTkFont(size=48)
        )
        icon_label.pack(pady=(0, 10))

        # Confirmation message
        message_label = ctk.CTkLabel(
            content_frame,
            text="Are you sure you want to logout?",
            font=self.subheader_font
        )
        message_label.pack(pady=10)

        # Buttons frame
        button_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(10, 0))

        # Cancel button
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            font=self.body_font,
            fg_color=self.neutral_color,
            hover_color="#374151",
            corner_radius=8,
            height=40,
            width=100,
            command=confirm_dialog.destroy
        )
        cancel_btn.pack(side="left", padx=(0, 10))

        # Function to handle logout
        def do_logout():
            confirm_dialog.destroy()
            try:
                # Start the login.py script
                python_executable = sys.executable
                subprocess.Popen([python_executable, "login.py"])

                # Exit the current application
                self.parent.after_cancel("all")
                self.parent.quit()
                self.parent.destroy()
                sys.exit()
            except Exception as e:
                self.show_notification(f"Could not restart the application: {str(e)}", "error")

        # Logout button
        logout_btn = ctk.CTkButton(
            button_frame,
            text="Logout",
            font=self.body_font,
            fg_color=self.danger_color,
            hover_color="#B91C1C",
            corner_radius=8,
            height=40,
            width=100,
            command=do_logout
        )
        logout_btn.pack(side="right")