import tkinter as tk
import tkinter.messagebox as tkmb
import re

from PIL import Image, ImageTk, ImageDraw
import json
import os
import base64

from cryptography.fernet import Fernet
from utils import decrypt_data, generate_key
from login import MASTER_KEY
from tkinter import filedialog, simpledialog, messagebox

class SettingsPage(tk.Frame):
    def __init__(self, parent, current_user_key, bg="#F1F1F1"):
        super().__init__(parent, bg=bg)
        self.USER_DATA_FILE = "users.json"
        self.SAVE_DIRECTORY = "user_data_directory"
        self.MASTER_KEY = MASTER_KEY
        self.parent = parent
        self.current_user_key = current_user_key
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
        self.grid(row=0, column=0, sticky="nsew")

    def decode_base64(self, data):
        return base64.b64decode(data).decode('utf-8')

    def load_user_data(self):
        try:
            with open('user_data_directory\\users.json', 'r') as file:
                data = json.load(file)
                print("User  data loaded successfully:", data)

                # Decrypt the email and any other necessary fields
                for user_key, user_info in data.items():
                    if 'email' in user_info:
                        # Decrypt the email using the existing decrypt_data function
                        encrypted_email = user_info['email']
                        # Assuming you have a way to get the encryption key
                        salt = user_key.encode()  # Use the username as the salt
                        key = generate_key(MASTER_KEY.decode(), salt)  # Generate the key
                        user_info['email'] = decrypt_data(base64.b64decode(encrypted_email), key)  # Decrypt the email

                    if 'designation' in user_info:
                        encrypted_designation = user_info['designation']
                        user_info['designation'] = decrypt_data(base64.b64decode(encrypted_designation), key)

                return data
        except FileNotFoundError:
            print("User  data file not found.")
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


                user_data_encrypted[username] = {
                    'password': encrypted_password_b64,
                    'email': encrypted_email_b64,
                    'designation': encrypted_designation_b64
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
        settingslb = tk.Label(self, text="SETTINGS", font=("Arial", 25, "bold"))
        settingslb.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="w")  # Align to the left

        # Load and display user image
        self.display_user_image()  # Call the updated method to load the specific image

        # Display user details

        username = self.current_user_key  # Use the current user key as the username
        username_label = tk.Label(self, text="Username:", font=("Arial", 14))
        username_label.grid(row=2, column=0, padx=20, pady=5, sticky="e")  # Align label to the right
        username_value = tk.Label(self, text=username, font=("Arial", 14))
        username_value.grid(row=2, column=1, padx=20, pady=5, sticky="w")  # Align value to the left

        email = self.current_user_data.get('email', '')
        email_label = tk.Label(self, text="Email:", font=("Arial", 14))
        email_label.grid(row=3, column=0, padx=20, pady=5, sticky="e")  # Align label to the right
        email_value = tk.Label(self, text=email, font=("Arial", 14))
        email_value.grid(row=3, column=1, padx=20, pady=5, sticky="w")  # Align value to the left

        designation = self.current_user_data.get('designation', '')
        designation_label = tk.Label(self, text="Designation:", font=("Arial", 14))
        designation_label.grid(row=4, column=0, padx=20, pady=5, sticky="e")  # Align label to the right
        designation_value = tk.Label(self, text=designation, font=("Arial", 14))
        designation_value.grid(row=4, column=1, padx=20, pady=5, sticky="w")  # Align value to the left

        upload_button = tk.Button(self, text="Upload Profile Picture", command=self.upload_image)
        upload_button.grid(row=5, column=0, columnspan=2, padx=20, pady=10)

        change_password_button = tk.Button(self, text="Change Password", command=self.change_password)
        change_password_button.grid(row=5, column=3, padx=20, pady=10)

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
            encrypted_password = self.current_user_data['password']
            decrypted_password = decrypt_data(base64.b64decode(encrypted_password), key)  # Decrypt
            decrypted_password_talaga = decrypt_data(base64.b64decode(decrypted_password), key)
            # Debugging: Print key and decrypted password
            print(f"[DEBUG] Decryption Key: {MASTER_KEY.decode()}")
            print(f"[DEBUG] Decrypted Password: {decrypted_password_talaga}")

        except Exception as e:
            messagebox.showerror("Error", f"Decryption failed: {e}")
            return

        # Step 4: Validate current password
        if current_password != decrypted_password_talaga:
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

    def display_user_image(self):
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
        self.user_image = ImageTk.PhotoImage(img)

        # Display the image
        image_label = tk.Label(self, image=self.user_image, bg=self['bg'])
        image_label.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="n")  # Center the image

    def configure_grid(self):
        # Configure grid weights to allow proper resizing
        self.grid_rowconfigure(1, weight=1)  # Row for image
        self.grid_rowconfigure(2, weight=1)  # Row for username
        self.grid_rowconfigure(3, weight=1)  # Row for email
        self.grid_rowconfigure(4, weight=1)  # Row for save button
        self.grid_columnconfigure(0, weight=1)  # Center column 0
        self.grid_columnconfigure(1, weight=1)  # Center column 1

