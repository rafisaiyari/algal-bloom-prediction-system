import customtkinter as ctk
import tkinter as tk  # Add this import for tk.END and other references
import tkinter.messagebox as tkmb  # Still using tkinter messagebox
import re
import sys
import subprocess

from PIL import Image, ImageTk, ImageDraw
import json
import os
import base64

from cryptography.fernet import Fernet
from utils import decrypt_data, generate_key
from login import MASTER_KEY
from tkinter import filedialog, simpledialog, messagebox


class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, current_user_key, current_user_type="regular", bg_color=None):
        super().__init__(parent, fg_color=bg_color or "transparent")
        self.USER_DATA_FILE = "users.json"
        self.SAVE_DIRECTORY = "user_data_directory"
        self.MASTER_KEY = MASTER_KEY
        self.parent = parent
        self.current_user_key = current_user_key
        self.current_user_type = current_user_type
        self.user_data = self.load_user_data()  # Load user data from JSON
        self.current_user_data = self.user_data.get(current_user_key, {})
        self.create_widgets()
        self.configure_grid()

    def encrypt_data(self, data, encryption_key):
        f = Fernet(encryption_key)
        encrypted_data = f.encrypt(data.encode())
        return encrypted_data

    def show(self):
        print(f"Current User Key in SettingsPage: {self.current_user_key}")
        print(f"Current User Type in SettingsPage: {self.current_user_type}")
        self.grid(row=0, column=0, sticky="nsew")

    def decode_base64(self, data):
        return base64.b64decode(data).decode('utf-8')

    def load_user_data(self):
        try:
            with open('user_data_directory\\users.json', 'r') as file:
                data = json.load(file)
                print("User data loaded successfully:", data)

                # Decrypt the data fields
                decrypted_data = {}
                for user_key, user_info in data.items():
                    # Generate the encryption key
                    salt = user_key.encode()  # Use the username as the salt
                    key = generate_key(MASTER_KEY.decode(), salt)  # Generate the key

                    decrypted_user = {}
                    try:
                        # Decrypt the standard fields
                        if 'email' in user_info:
                            decrypted_user['email'] = decrypt_data(base64.b64decode(user_info['email']), key)

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

    def save_user_data(self, data):
        try:
            user_data_encrypted = {}
            for username, user_data in data.items():
                salt = username.encode()
                key = generate_key(self.MASTER_KEY.decode(), salt)

                encrypted_password = self.encrypt_data(user_data['password'], key)
                encrypted_password_b64 = base64.b64encode(encrypted_password).decode()

                encrypted_email = self.encrypt_data(user_data['email'], key)
                encrypted_email_b64 = base64.b64encode(encrypted_email).decode()

                encrypted_designation = self.encrypt_data(user_data['designation'], key)
                encrypted_designation_b64 = base64.b64encode(encrypted_designation).decode()

                # Encrypt user type
                user_type = user_data.get('user_type', 'regular')
                encrypted_user_type = self.encrypt_data(user_type, key)
                encrypted_user_type_b64 = base64.b64encode(encrypted_user_type).decode()

                user_data_encrypted[username] = {
                    'password': encrypted_password_b64,
                    'email': encrypted_email_b64,
                    'designation': encrypted_designation_b64,
                    'user_type': encrypted_user_type_b64
                }

            json_string = json.dumps(user_data_encrypted)

            filepath = os.path.join(self.SAVE_DIRECTORY, self.USER_DATA_FILE)
            with open(filepath, "w") as file:
                file.write(json_string)
            return True
        except Exception as e:
            tkmb.showerror(title="Error", message=f"Error saving user data: {e}")
            return False

    def create_widgets(self):
        settingslb = ctk.CTkLabel(self, text="SETTINGS", font=("Arial", 25, "bold"))
        settingslb.grid(row=0, column=0, columnspan=3, padx=20, pady=20, sticky="w")  # Align to the left

        # Create a frame to hold profile picture and user info side by side
        profile_frame = ctk.CTkFrame(self, fg_color="transparent")
        profile_frame.grid(row=1, column=0, columnspan=3, padx=20, pady=10, sticky="w")

        # Load and display user image in the profile frame (left side)
        self.display_user_image(profile_frame)

        # Create a frame for user details (right side of profile picture)
        user_info_frame = ctk.CTkFrame(profile_frame, fg_color="transparent")
        user_info_frame.pack(side="right", padx=20, fill="both", expand=True)

        # Display user details in the user_info_frame
        username = self.current_user_key  # Use the current user key as the username
        username_label = ctk.CTkLabel(user_info_frame, text="Username:", font=("Arial", 14))
        username_label.grid(row=0, column=0, padx=5, pady=10, sticky="w")
        username_value = ctk.CTkLabel(user_info_frame, text=username, font=("Arial", 14))
        username_value.grid(row=0, column=1, padx=5, pady=10, sticky="w")

        email = self.current_user_data.get('email', '')
        email_label = ctk.CTkLabel(user_info_frame, text="Email:", font=("Arial", 14))
        email_label.grid(row=1, column=0, padx=5, pady=10, sticky="w")
        email_value = ctk.CTkLabel(user_info_frame, text=email, font=("Arial", 14))
        email_value.grid(row=1, column=1, padx=5, pady=10, sticky="w")

        designation = self.current_user_data.get('designation', '')
        designation_label = ctk.CTkLabel(user_info_frame, text="Designation:", font=("Arial", 14))
        designation_label.grid(row=2, column=0, padx=5, pady=10, sticky="w")
        designation_value = ctk.CTkLabel(user_info_frame, text=designation, font=("Arial", 14))
        designation_value.grid(row=2, column=1, padx=5, pady=10, sticky="w")

        # Display user type
        user_type = self.current_user_data.get('user_type', 'regular')
        user_type_label = ctk.CTkLabel(user_info_frame, text="User Type:", font=("Arial", 14))
        user_type_label.grid(row=3, column=0, padx=5, pady=10, sticky="w")
        user_type_value = ctk.CTkLabel(user_info_frame, text=user_type.upper(), font=("Arial", 14))
        user_type_value.grid(row=3, column=1, padx=5, pady=10, sticky="w")

        # Add buttons below the profile information
        buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        buttons_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=10, sticky="w")

        upload_button = ctk.CTkButton(buttons_frame, text="Upload Profile Picture", command=self.upload_image)
        upload_button.pack(side="left", padx=10)

        change_password_button = ctk.CTkButton(buttons_frame, text="Change Password", command=self.change_password)
        change_password_button.pack(side="left", padx=10)

        # Add logout button
        logout_button = ctk.CTkButton(buttons_frame, text="Logout", command=self.logout, fg_color="#FF5555",
                                      hover_color="#E04040")
        logout_button.pack(side="left", padx=10)

        # User management section for master users
        if self.current_user_type == "master":
            self.create_user_management_widgets()

    def logout(self):
        # Ask for confirmation before exiting
        if messagebox.askyesno("Logout", "Are you sure you want to logout?"):
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
                messagebox.showerror("Error", f"Could not restart the application: {str(e)}")

    def create_user_management_widgets(self):
        # Create a frame for user management
        user_mgmt_frame = ctk.CTkFrame(self)
        user_mgmt_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=20, sticky="ew")

        # Add a label to the frame
        mgmt_title = ctk.CTkLabel(user_mgmt_frame, text="User Management", font=("Arial", 14, "bold"))
        mgmt_title.pack(anchor="w", padx=10, pady=10)

        # Create a container for the content
        mgmt_content = ctk.CTkFrame(user_mgmt_frame, fg_color="transparent")
        mgmt_content.pack(fill="both", expand=True, padx=10, pady=5)

        # List all users
        users_label = ctk.CTkLabel(mgmt_content, text="Select User:", font=("Arial", 12))
        users_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Create a listbox of users - CTk doesn't have a direct listbox equivalent, so we'll use a custom widget or Tkinter's
        # For now we'll use Tkinter's Listbox since it has better built-in selection handling
        self.users_listbox = tk.Listbox(mgmt_content, width=30, height=10, font=("Arial", 10))
        self.users_listbox.grid(row=1, column=0, rowspan=4, padx=5, pady=5, sticky="ns")

        # Populate the listbox with users
        self.populate_users_listbox()

        # User type selection
        user_type_label = ctk.CTkLabel(mgmt_content, text="Change User Type:", font=("Arial", 12))
        user_type_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.user_type_var = ctk.StringVar(value="regular")
        regular_radio = ctk.CTkRadioButton(mgmt_content, text="Regular User", variable=self.user_type_var,
                                           value="regular")
        regular_radio.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        superuser_radio = ctk.CTkRadioButton(mgmt_content, text="Super User", variable=self.user_type_var,
                                             value="superuser")
        superuser_radio.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        master_radio = ctk.CTkRadioButton(mgmt_content, text="Master User", variable=self.user_type_var, value="master")
        master_radio.grid(row=3, column=2, padx=5, pady=5, sticky="w")

        # Button to apply changes
        apply_button = ctk.CTkButton(mgmt_content, text="Apply Changes", command=self.change_user_type)
        apply_button.grid(row=4, column=2, padx=5, pady=5, sticky="e")

        # Button to refresh user list
        refresh_button = ctk.CTkButton(mgmt_content, text="Refresh", command=self.populate_users_listbox)
        refresh_button.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Listbox selection event
        self.users_listbox.bind('<<ListboxSelect>>', self.on_user_select)

    def populate_users_listbox(self):
        """Populate the listbox with users and their types"""
        self.users_listbox.delete(0, tk.END)  # Clear the listbox

        for username, user_data in self.user_data.items():
            user_type = user_data.get('user_type', 'regular')
            self.users_listbox.insert(tk.END, f"{username} ({user_type})")

    def on_user_select(self, event):
        """Handle user selection from listbox"""
        if not self.users_listbox.curselection():
            return

        # Get selected item (username and type)
        selected_item = self.users_listbox.get(self.users_listbox.curselection())

        # Extract username and user type
        username = selected_item.split(" (")[0]
        current_type = selected_item.split("(")[1].rstrip(")")

        # Set the radio button to match the current user type
        self.user_type_var.set(current_type)

    def change_user_type(self):
        """Change the selected user's type"""
        if not self.users_listbox.curselection():
            messagebox.showwarning("Warning", "Please select a user first.")
            return

        # Get selected user
        selected_item = self.users_listbox.get(self.users_listbox.curselection())
        username = selected_item.split(" (")[0]

        # Don't allow changing your own user type
        if username == self.current_user_key:
            messagebox.showwarning("Warning", "You cannot change your own user type.")
            return

        # Get selected user type
        new_user_type = self.user_type_var.get()

        # Confirm change
        confirm = messagebox.askyesno("Confirm",
                                      f"Are you sure you want to change {username}'s user type to {new_user_type}?")
        if not confirm:
            return

        # Update user type
        if username in self.user_data:
            self.user_data[username]['user_type'] = new_user_type

            # Save changes
            if self.save_user_data(self.user_data):
                messagebox.showinfo("Success", f"{username}'s user type updated to {new_user_type}.")
                self.populate_users_listbox()  # Refresh the list
            else:
                messagebox.showerror("Error", "Failed to save changes.")

    def change_password(self):
        def is_valid_password(password):
            pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+.,])[A-Za-z\d!@#$%^&*()_+.,]{8,}$"
            return bool(re.match(pattern, password))

        # Step 1: Get current password
        current_password = simpledialog.askstring("Current Password", "Enter your current password:",
                                                  parent=self.parent)
        if not current_password:
            messagebox.showwarning("Warning", "Current password cannot be empty.")
            return

        # Step 2: Generate decryption key
        salt = self.current_user_key.encode()  # Salt = username
        key = generate_key(MASTER_KEY.decode(), salt)  # Must match key used during signup

        try:
            # Step 3: Decrypt stored password
            stored_password = self.current_user_data.get('password', '')

            # Debugging: Print key and decrypted password
            print(f"[DEBUG] Decryption Key: {MASTER_KEY.decode()}")
            print(f"[DEBUG] Stored Password: {stored_password}")

        except Exception as e:
            messagebox.showerror("Error", f"Decryption failed: {e}")
            return

        # Step 4: Validate current password
        if current_password != stored_password:
            messagebox.showwarning("Warning", "Current password is incorrect.")
            return

        # Step 5: Get and validate new password
        new_password = simpledialog.askstring("Change Password", "Enter new password:", parent=self.parent)
        if not new_password:
            messagebox.showwarning("Warning", "New password cannot be empty.")
            return

        if not is_valid_password(new_password):
            messagebox.showwarning("Warning",
                                   "Password must have:\n- 8+ chars"
                                   "\n- Uppercase & lowercase\n- Number\n- Special character")
            return

        # Step 6: Confirm new password
        confirm_password = simpledialog.askstring("Confirm Password",
                                                  "Confirm new password:", parent=self.parent)
        if new_password != confirm_password:
            messagebox.showwarning("Warning", "Passwords do not match.")
            return

        # Step 7: Update and save
        try:
            self.current_user_data['password'] = new_password  # Store plaintext temporarily
            self.user_data[self.current_user_key] = self.current_user_data

            if self.save_user_data(self.user_data):  # save_user_data will encrypt it
                messagebox.showinfo("Success", "Password updated successfully!")
            else:
                messagebox.showerror("Error", "Failed to save password.")
        except Exception as e:
            messagebox.showerror("Error", f"Update failed: {str(e)}")

    def upload_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(title="Select Profile Picture",
                                               filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # Save the selected image with the user's key as the filename
            new_image_path = os.path.join('user_data_directory', f"{self.current_user_key}_profile_image.png")
            img = Image.open(file_path)
            img = img.resize((300, 300), Image.LANCZOS)  # Resize image to fit
            img.save(new_image_path)  # Save the new image

            # Update the displayed image
            self.display_user_image()

    def display_user_image(self, container=None):
        # If no container is provided, use self as the container
        if container is None:
            container = self

        # Construct the user-specific image path
        user_image_path = os.path.join('user_data_directory', f"{self.current_user_key}_profile_image.png")
        default_image_path = os.path.join('user_data_directory',
                                          'default_profile_image.jpg')  # Path to the default image

        # Check if the user-specific image exists
        if os.path.exists(user_image_path):
            image_path = user_image_path
        elif os.path.exists(default_image_path):
            image_path = default_image_path
        else:
            print("No profile image found for user and no default image available.")
            return

        # Load and process the image
        img = Image.open(image_path)
        img = img.resize((300, 300), Image.LANCZOS)  # Resize image to fit

        # Create a circular mask
        mask = Image.new('L', (300, 300), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, 300, 300), fill=255)

        # Apply mask to image
        img.putalpha(mask)
        self.user_image = ctk.CTkImage(light_image=img, dark_image=img, size=(300, 300))

        # Create image frame
        image_frame = ctk.CTkFrame(container, fg_color="transparent")
        image_frame.pack(side="left", padx=10)

        # Display the image in the frame
        image_label = ctk.CTkLabel(image_frame, image=self.user_image, text="")
        image_label.pack(padx=10, pady=10)

    def configure_grid(self):
        # Configure grid weights to allow proper resizing
        self.grid_rowconfigure(0, weight=0)  # Settings header
        self.grid_rowconfigure(1, weight=1)  # Profile frame row
        self.grid_rowconfigure(2, weight=0)  # Buttons row
        self.grid_rowconfigure(3, weight=1)  # User management row (if visible)

        self.grid_columnconfigure(0, weight=1)  # Left column
        self.grid_columnconfigure(1, weight=2)  # Center column
        self.grid_columnconfigure(2, weight=1)  # Right column