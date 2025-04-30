import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import re
import sys
import subprocess
import os
import json
import base64
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
from cryptography.fernet import Fernet
from utils import decrypt_data, generate_key
from login import MASTER_KEY


class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, current_user_key, current_user_type="regular", bg_color=None):
        super().__init__(parent, fg_color=bg_color or "#FFFFFF")

        # Constants
        self.USER_DATA_FILE = "users.json"
        self.SAVE_DIRECTORY = "user_data_directory"
        self.MASTER_KEY = MASTER_KEY

        # Instance variables
        self.parent = parent
        self.current_user_key = current_user_key
        self.current_user_type = current_user_type

        # Modern color scheme
        self.primary_color = "#2563EB"  # Blue
        self.primary_light = "#DBEAFE"  # Light blue background
        self.primary_dark = "#1E40AF"  # Dark blue for hover states
        self.danger_color = "#DC2626"  # Red for dangerous actions
        self.success_color = "#10B981"  # Green for success indicators
        self.neutral_color = "#4B5563"  # Gray for neutral actions
        self.dark_text = "#1F2937"  # Nearly black for text
        self.light_text = "#F9FAFB"  # Nearly white for text on dark backgrounds
        self.border_color = "#E5E7EB"  # Light gray for borders
        self.bg_color = bg_color or "#FFFFFF"  # Match main app bg color
        self.card_bg = "#FFFFFF"  # White for card backgrounds

        # Font settings
        self.header_font = ctk.CTkFont(family="Segoe UI", size=22, weight="bold")
        self.subheader_font = ctk.CTkFont(family="Segoe UI", size=16, weight="bold")
        self.body_font = ctk.CTkFont(family="Segoe UI", size=13)
        self.small_font = ctk.CTkFont(family="Segoe UI", size=11)
        self.button_font = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")

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

    def load_user_data(self):
        """Load and decrypt user data from JSON file"""
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

    def change_user_type(self):
        """Change the selected user's type"""
        if not self.users_listbox.curselection():
            self.show_notification("Please select a user first", "warning")
            return

        # Get selected user
        selected_item = self.users_listbox.get(self.users_listbox.curselection())
        username = selected_item.split(" (")[0]

        # Find the user object
        user_to_update = None
        for user in self.users:
            if user.username == username:
                user_to_update = user
                break

        if not user_to_update:
            self.show_notification(f"User '{username}' not found in the system.", "error")
            return

        # Create a dialog to get the new user type
        new_type_dialog = tk.Toplevel(self.master)
        new_type_dialog.title("Change User Type")

        new_type_label = tk.Label(new_type_dialog, text="Select new user type:")
        new_type_label.pack(pady=5)

        user_types = ["admin", "regular"]  # Assuming you have these user types
        self.new_type_var = tk.StringVar(new_type_dialog)
        self.new_type_var.set(user_to_update.user_type)  # Set current type as default

        new_type_dropdown = tk.OptionMenu(new_type_dialog, self.new_type_var, *user_types)
        new_type_dropdown.pack(pady=5)

        change_button = tk.Button(new_type_dialog, text="Change Type", command=self._update_user_type)
        change_button.pack(pady=10)

        # Store the user to update in an instance variable so _update_user_type can access it
        self.user_to_update = user_to_update
        self.new_type_dialog = new_type_dialog

    def _update_user_type(self):
        """Updates the user type after the user confirms in the dialog."""
        if hasattr(self, 'user_to_update') and hasattr(self, 'new_type_var'):
            new_user_type = self.new_type_var.get()
            self.user_to_update.user_type = new_user_type
            self.show_notification(f"User '{self.user_to_update.username}' type changed to '{new_user_type}'.", "info")
            self._populate_user_list()  # Refresh the user list in the UI
            self.new_type_dialog.destroy()
            # Optionally, save the updated user data to your storage (e.g., file, database)
            # self._save_user_data()
        else:
            self.show_notification("Error updating user type.", "error")

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
        """Create the main UI framework - called when page is shown"""
        # Clear existing widgets if any
        if self.main_container:
            self.main_container.destroy()

        # Get current window dimensions
        self.update_idletasks()  # Ensure geometry info is updated

        # Use parent's dimensions rather than self to avoid recursion issues
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Use reasonable defaults if dimensions are too small
        if parent_width < 200:
            parent_width = 800
        if parent_height < 200:
            parent_height = 600

        # Cache these values
        self.last_width = parent_width
        self.last_height = parent_height

        print(f"Setting up UI with dimensions: {parent_width}x{parent_height}")

        # Create scrollable container with proper dimensions - fixed width to prevent expansion
        self.main_container = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_fg_color=self.primary_light,
            scrollbar_button_color=self.primary_color,
            scrollbar_button_hover_color=self.primary_dark,
            width=min(parent_width - 40, 1200),  # Add maximum width to prevent infinite expansion
            height=parent_height - 40
        )
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Important: Configure the column weight in main_container
        self.main_container.grid_columnconfigure(0, weight=1)

        # Create sections
        self.create_header_section()
        self.create_profile_section()
        self.create_actions_section()

        # Add user management section if user is a master
        if self.current_user_type == "master":
            self.create_user_management_section()

        # Set flag to track that UI is initialized
        self.is_ui_initialized = True

        # Bind to configure event to handle resizing
        self.bind("<Configure>", self.on_resize)

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
        header_frame.grid_columnconfigure(0, weight=1)  # Important: configure weight

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
        self.image_container = ctk.CTkFrame(profile_frame, fg_color="transparent", width=160, height=200)
        self.image_container.grid(row=1, column=0, padx=20, pady=20, sticky="n")
        self.image_container.grid_propagate(False)  # Important to maintain size
        self.display_user_image(self.image_container)

        # User information (right side)
        info_frame = ctk.CTkFrame(profile_frame, fg_color="transparent")
        info_frame.grid(row=1, column=1, padx=(0, 20), pady=20, sticky="nsew")

        # User role badge
        user_type = self.current_user_data.get('user_type', 'regular')
        badge_color = self.primary_color if user_type == "regular" else "#9333EA" if user_type == "master" else "#F97316"

        badge_frame = ctk.CTkFrame(info_frame, fg_color=badge_color, corner_radius=5)
        badge_frame.grid(row=0, column=0, sticky="w")

        badge_label = ctk.CTkLabel(
            badge_frame,
            text=f" {user_type.upper()} USER ",
            font=self.small_font,
            text_color=self.light_text
        )
        badge_label.pack(padx=10, pady=5)

        # Create info fields with a more modern look
        self.add_info_field(info_frame, 1, "Username", self.current_user_key)

        self.add_info_field(info_frame, 3, "Designation", self.current_user_data.get('designation', ''))

        # Account creation date (example static data)
        self.add_info_field(info_frame, 4, "Account Status", "Active", "#10B981")

    def add_info_field(self, parent, row, label_text, value_text, value_color=None):
        """Helper to create a consistent field layout"""
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

        # Value with prominent color
        value = ctk.CTkLabel(
            field_frame,
            text=value_text,
            font=self.body_font,
            text_color=value_color or self.dark_text,
            anchor="w"
        )
        value.pack(anchor="w")

    def create_actions_section(self):
        """Create action buttons in a card layout"""
        # Container for all action cards
        actions_container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        actions_container.grid(row=2, column=0, sticky="ew", padx=30, pady=(0, 20))

        # Configure grid layout for cards (2 columns)
        actions_container.grid_columnconfigure(0, weight=1)
        actions_container.grid_columnconfigure(1, weight=1)

        # Add action cards
        self.create_action_card(
            actions_container, 0, 0,
            "📷", "Change Profile Picture",
            "Update your profile image",
            self.upload_image,
            self.primary_color
        )

        self.create_action_card(
            actions_container, 0, 1,
            "🔒", "Update Password",
            "Change your account password",
            self.change_password,
            self.primary_color
        )

        # Add logout in a separate card that spans both columns
        logout_card = ctk.CTkFrame(
            actions_container,
            fg_color=self.card_bg,
            corner_radius=15,
            border_width=1,
            border_color=self.border_color
        )
        logout_card.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        # Center the content in the logout card
        logout_card.grid_columnconfigure(0, weight=1)

        logout_button = ctk.CTkButton(
            logout_card,
            text="Sign Out",
            font=self.button_font,
            fg_color=self.danger_color,
            hover_color="#B91C1C",  # Darker red on hover
            corner_radius=8,
            height=45,
            width=200,
            command=self.logout
        )
        logout_button.grid(row=0, column=0, padx=20, pady=20)

    def create_action_card(self, parent, row, col, icon, title, description, command, color):
        """Create a card for a single action"""
        card = ctk.CTkFrame(
            parent,
            fg_color=self.card_bg,
            corner_radius=15,
            border_width=1,
            border_color=self.border_color
        )
        card.grid(row=row, column=col, sticky="ew", padx=10, pady=10)

        # Card content frame
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # Icon
        icon_label = ctk.CTkLabel(
            content,
            text=icon,
            font=ctk.CTkFont(family="Segoe UI", size=32),
            text_color=color
        )
        icon_label.pack(anchor="w")

        # Title
        title_label = ctk.CTkLabel(
            content,
            text=title,
            font=self.subheader_font,
            text_color=self.dark_text,
            anchor="w"
        )
        title_label.pack(anchor="w", pady=(5, 0))

        # Description
        desc_label = ctk.CTkLabel(
            content,
            text=description,
            font=self.small_font,
            text_color=self.neutral_color,
            anchor="w"
        )
        desc_label.pack(anchor="w", pady=(0, 10))

        # Button
        button = ctk.CTkButton(
            content,
            text="Open",
            font=self.small_font,
            fg_color=color,
            hover_color=self.primary_dark,
            corner_radius=8,
            height=30,
            width=80,
            command=command
        )
        button.pack(anchor="w")

    def create_user_management_section(self):
        """Create the user management section for master users"""
        # Admin section container
        admin_frame = ctk.CTkFrame(
            self.main_container,
            fg_color=self.card_bg,
            corner_radius=15,
            border_width=1,
            border_color=self.border_color
        )
        admin_frame.grid(row=3, column=0, sticky="ew", padx=30, pady=(0, 30))

        # Admin section header with special styling
        header_bg = ctk.CTkFrame(
            admin_frame,
            fg_color="#9333EA",  # Purple for admin section
            corner_radius=12,
            height=60
        )
        header_bg.pack(fill="x", padx=2, pady=(2, 20))

        header_content = ctk.CTkFrame(header_bg, fg_color="transparent")
        header_content.pack(fill="both", expand=True, padx=20)

        admin_icon = ctk.CTkLabel(
            header_content,
            text="👑",
            font=ctk.CTkFont(size=24)
        )
        admin_icon.pack(side="left", padx=(0, 10))

        admin_title = ctk.CTkLabel(
            header_content,
            text="User Management Console",
            font=self.subheader_font,
            text_color=self.light_text
        )
        admin_title.pack(side="left")

        # Create the user management interface
        self.create_user_management_interface(admin_frame)

    def create_user_management_interface(self, parent):
        """Create the user management UI components"""
        # Content container with padding
        content_frame = ctk.CTkFrame(parent, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Configure columns: 40% for user list, 60% for user details
        content_frame.grid_columnconfigure(0, weight=2)
        content_frame.grid_columnconfigure(1, weight=3)

        # Left side - User list with search
        list_frame = ctk.CTkFrame(content_frame, fg_color=self.primary_light, corner_radius=10)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        # Search bar
        search_frame = ctk.CTkFrame(list_frame, fg_color="transparent")
        search_frame.pack(fill="x", padx=15, pady=15)

        search_icon = ctk.CTkLabel(search_frame, text="🔍", font=self.body_font)
        search_icon.pack(side="left", padx=(0, 5))

        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", self.filter_users)

        search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Search users...",
            textvariable=self.search_var,
            height=36,
            corner_radius=8,
            border_width=0,
            fg_color="#FFFFFF"
        )
        search_entry.pack(side="left", fill="x", expand=True)

        # User list with custom styling
        list_container = ctk.CTkFrame(list_frame, fg_color="transparent")
        list_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Create a custom frame with white background for the listbox
        listbox_bg = ctk.CTkFrame(list_container, fg_color="#FFFFFF", corner_radius=8)
        listbox_bg.pack(fill="both", expand=True)

        # Custom style for the listbox
        self.users_listbox = tk.Listbox(
            listbox_bg,
            font=("Segoe UI", 12),
            borderwidth=0,
            highlightthickness=0,
            selectbackground=self.primary_color,
            selectforeground="#FFFFFF",
            activestyle="none",
            background="#FFFFFF"
        )
        self.users_listbox.pack(side="left", fill="both", expand=True, padx=2, pady=2)

        # Modern scrollbar
        scrollbar = ctk.CTkScrollbar(
            listbox_bg,
            command=self.users_listbox.yview,
            fg_color="#FFFFFF",
            button_color=self.primary_color,
            button_hover_color=self.primary_dark
        )
        scrollbar.pack(side="right", fill="y")
        self.users_listbox.config(yscrollcommand=scrollbar.set)

        # Populate the listbox
        self.populate_users_listbox()

        # Right side - User details and controls
        details_frame = ctk.CTkFrame(content_frame, fg_color=self.card_bg, corner_radius=10)
        details_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)

        # Selected user header
        self.selected_header = ctk.CTkLabel(
            details_frame,
            text="Select a user to manage",
            font=self.subheader_font,
            anchor="w"
        )
        self.selected_header.pack(anchor="w", padx=20, pady=(20, 10))

        # User type selection with better styling
        self.type_selector = ctk.CTkFrame(details_frame, fg_color="transparent")
        self.type_selector.pack(fill="x", padx=20, pady=10)

        self.user_type_var = ctk.StringVar(value="regular")

        # Create type selections with better visual indicators
        self.create_type_option("regular", "Regular User",
                                "Basic access to system features",
                                self.primary_color)

        self.create_type_option("superuser", "Super User",
                                "Advanced access with elevated privileges",
                                "#F97316")  # Orange

        self.create_type_option("master", "Master User",
                                "Full administrative control",
                                "#9333EA")  # Purple

        # Create a bottom action bar with buttons
        action_bar = ctk.CTkFrame(details_frame, fg_color="transparent")
        action_bar.pack(fill="x", padx=20, pady=(20, 20), side="bottom")

        # Apply button
        apply_btn = ctk.CTkButton(
            action_bar,
            text="Apply Changes",
            font=self.button_font,
            fg_color=self.success_color,
            hover_color="#059669",  # Darker green
            corner_radius=8,
            height=40,
            command=self.change_user_type
        )
        apply_btn.pack(side="right", padx=(10, 0))

        # Refresh button
        refresh_btn = ctk.CTkButton(
            action_bar,
            text="Refresh List",
            font=self.button_font,
            fg_color=self.neutral_color,
            hover_color="#374151",  # Darker gray
            corner_radius=8,
            height=40,
            command=self.populate_users_listbox
        )
        refresh_btn.pack(side="right")

        # Listbox selection event
        self.users_listbox.bind('<<ListboxSelect>>', self.on_user_select)

    def create_type_option(self, type_value, title, description, color):
        """Create a radio option with modern styling"""
        option_frame = ctk.CTkFrame(self.type_selector, fg_color="transparent")
        option_frame.pack(fill="x", pady=8)

        # Radio button
        radio = ctk.CTkRadioButton(
            option_frame,
            text="",
            variable=self.user_type_var,
            value=type_value,
            fg_color=color,
            border_color=self.neutral_color
        )
        radio.pack(side="left", padx=(0, 10))

        # Create text container
        text_frame = ctk.CTkFrame(option_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="x", expand=True)

        # Title
        title_label = ctk.CTkLabel(
            text_frame,
            text=title,
            font=self.body_font,
            text_color=self.dark_text,
            anchor="w"
        )
        title_label.pack(anchor="w")

        # Description
        desc_label = ctk.CTkLabel(
            text_frame,
            text=description,
            font=self.small_font,
            text_color=self.neutral_color,
            anchor="w"
        )
        desc_label.pack(anchor="w")

        # Add a visual indicator of the user type color
        indicator = ctk.CTkFrame(
            option_frame,
            fg_color=color,
            width=5,
            corner_radius=2
        )
        indicator.pack(side="right", fill="y", padx=(10, 0))

    def filter_users(self, *args):
        """Filter users list based on search text"""
        search_text = self.search_var.get().lower()
        self.populate_users_listbox(filter_text=search_text)

    def populate_users_listbox(self, filter_text=None):
        """Populate the listbox with users and their types"""
        self.users_listbox.delete(0, tk.END)

        sorted_users = sorted(self.user_data.items(), key=lambda x: x[0].lower())

        for username, user_data in sorted_users:
            if filter_text and filter_text not in username.lower():
                continue

            user_type = user_data.get('user_type', 'regular')
            self.users_listbox.insert(tk.END, f"{username} ({user_type})")

            # Color-code based on user type
            if user_type == "master":
                self.users_listbox.itemconfig(tk.END, fg="#9333EA")  # Purple for master
            elif user_type == "superuser":
                self.users_listbox.itemconfig(tk.END, fg="#F97316")  # Orange for superuser

    def on_user_select(self, event):
        """Handle user selection from listbox"""
        if not self.users_listbox.curselection():
            self.selected_header.configure(text="Select a user to manage")
            return

        # Get selected item (username and type)
        selected_item = self.users_listbox.get(self.users_listbox.curselection())

        # Extract username and user type
        username = selected_item.split(" (")[0]

        # Don't allow changing your own user type
        if username == self.current_user_key:
            self.show_notification("You cannot change your own user type", "warning")
            return

        # Get selected user type
        new_user_type = self.user_type_var.get()

        # Confirm change
        confirm = messagebox.askyesno(
            "Confirm Changes",
            f"Are you sure you want to change {username}'s user type to {new_user_type}?",
            icon=messagebox.WARNING
        )
        if not confirm:
            return

        # Update user type
        if username in self.user_data:
            self.user_data[username]['user_type'] = new_user_type

            # Save changes
            if self.save_user_data(self.user_data):
                self.show_notification(f"{username}'s user type updated to {new_user_type}", "success")
                self.populate_users_listbox()  # Refresh the list
            else:
                self.show_notification("Failed to save changes", "error")

    def show_notification(self, message, type="info"):
        """Show a notification with consistent styling"""
        icon = "ℹ️" if type == "info" else "✅" if type == "success" else "⚠️" if type == "warning" else "❌"

        if type == "error":
            messagebox.showerror("Error", f"{icon} {message}")
        elif type == "warning":
            messagebox.showwarning("Warning", f"{icon} {message}")
        elif type == "success":
            messagebox.showinfo("Success", f"{icon} {message}")
        else:
            messagebox.showinfo("Information", f"{icon} {message}")

    def display_user_image(self, container=None):
        """Display the user profile image with modern styling"""
        # If no container is provided, create a new one
        if container is None:
            container = ctk.CTkFrame(self, fg_color="transparent")
            container.pack(padx=10, pady=10)

        # Clear any existing widgets in the container
        for widget in container.winfo_children():
            widget.destroy()

        # Construct the user-specific image path
        user_image_path = os.path.join('user_data_directory', f"{self.current_user_key}_profile_image.png")
        default_image_path = os.path.join('user_data_directory', 'default_profile_image.jpg')

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
            img = img.resize((120, 120), Image.LANCZOS)  # Resize image

            # Create a circular mask
            mask = Image.new('L', (120, 120), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, 120, 120), fill=255)

            # Apply the mask to create a circular image
            img_rgba = img.convert("RGBA")
            circle_img = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
            circle_img.paste(img_rgba, (0, 0), mask)

            # Create a CTkImage from the circular image
            self.user_image = ctk.CTkImage(
                light_image=circle_img,
                dark_image=circle_img,
                size=(120, 120)
            )

            # Create layered frames for shadow effect
            shadow_frame = ctk.CTkFrame(
                container,
                fg_color="#E5E7EB",
                corner_radius=60,
                width=130,
                height=130
            )
            shadow_frame.pack(padx=5, pady=5)

            # Display the image with a white border
            image_bg = ctk.CTkFrame(
                shadow_frame,
                fg_color="#FFFFFF",
                corner_radius=60,
                width=126,
                height=126
            )
            image_bg.place(relx=0.5, rely=0.5, anchor="center")

            image_label = ctk.CTkLabel(
                image_bg,
                image=self.user_image,
                text="",
                width=120,
                height=120
            )
            image_label.place(relx=0.5, rely=0.5, anchor="center")

        except Exception as e:
            print(f"Error displaying user image: {e}")
            # Create a placeholder with initials if image fails
            self.create_initials_placeholder(container)

    def create_initials_placeholder(self, container):
        """Create a placeholder with user initials if no image is available"""
        # Get user initials
        initials = self.current_user_key[0].upper() if self.current_user_key else "U"

        # Create a frame for the circular background
        circle_frame = ctk.CTkFrame(
            container,
            fg_color=self.primary_color,
            width=120,
            height=120,
            corner_radius=60
        )
        circle_frame.pack(padx=5, pady=5)

        # Add the initials label
        initials_label = ctk.CTkLabel(
            circle_frame,
            text=initials,
            font=ctk.CTkFont(family="Segoe UI", size=50, weight="bold"),
            text_color="#FFFFFF"
        )
        initials_label.place(relx=0.5, rely=0.5, anchor="center")

        # Add change photo button
        change_photo_btn = ctk.CTkButton(
            container,
            text="Change Photo",
            font=self.small_font,
            fg_color=self.primary_color,
            hover_color=self.primary_dark,
            corner_radius=20,
            height=28,
            width=120,
            command=self.upload_image
        )
        change_photo_btn.pack(pady=(5, 0))

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
        password_dialog.geometry("400x350")
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
        confirm_entry.pack(fill="x", pady=(0, 20))

        # Error message label (hidden by default)
        error_var = ctk.StringVar(value="")
        error_label = ctk.CTkLabel(
            content_frame,
            textvariable=error_var,
            font=self.small_font,
            text_color=self.danger_color
        )
        error_label.pack(fill="x", pady=(0, 10))

        # Buttons frame
        button_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        button_frame.pack(fill="x")

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
            command=password_dialog.destroy
        )
        cancel_btn.pack(side="left", padx=(0, 10))

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

        # Change button
        change_btn = ctk.CTkButton(
            button_frame,
            text="Update Password",
            font=self.body_font,
            fg_color=self.primary_color,
            hover_color=self.primary_dark,
            corner_radius=8,
            height=40,
            width=150,
            command=do_change_password
        )
        change_btn.pack(side="right")

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

    def show(self):
        """Show the settings page in the parent's grid"""
        self.grid(row=0, column=0, sticky="nsew")

        # Force update of geometry
        self.update_idletasks()

        # Initialize UI if it hasn't been set up yet
        if not self.is_ui_initialized:
            self.setup_ui()
        else:
            # If already initialized, just update dimensions
            self.on_resize()

        print(f"Settings page shown with dimensions: {self.winfo_width()}x{self.winfo_height()}")


# Function to use this class in place of the original SettingsPage
def create_settings_page(parent, current_user_key, current_user_type="regular", bg_color=None):
    """Create and return a modern settings page instance"""
    return SettingsPage(parent, current_user_key, current_user_type, bg_color)
    current_type = selected_item.split("(")[1].rstrip(")")

    # Update the selected user header
    self.selected_header.configure(text=f"Managing User: {username}")

    # Set the radio button to match the current user type
    self.user_type_var.set(current_type)


