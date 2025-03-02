import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import json
import os
import base64
# Import the global variable from login.py
from globals import current_user_key
from utils import generate_key

current_user_key = 'rafi2'


class SettingsPage(tk.Frame):
    def __init__(self, parent, bg="#F1F1F1"):
        super().__init__(parent, bg=bg)
        self.parent = parent
        self.user_data = self.load_user_data()  # Load user data from JSON
        self.current_user_data = self.user_data.get(current_user_key, {})  # Access user data for the current user
        self.create_widgets()
        self.configure_grid()

    def show(self):
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
                        from login import decrypt_data, generate_key, MASTER_KEY
                        encrypted_email = user_info['email']
                        # Assuming you have a way to get the encryption key
                        salt = user_key.encode()  # Use the username as the salt
                        key = generate_key(MASTER_KEY.decode(), salt)  # Generate the key
                        user_info['email'] = decrypt_data(base64.b64decode(encrypted_email), key)  # Decrypt the email

                    if 'age' in user_info:
                        encrypted_age = user_info['age']
                        user_info['age'] = decrypt_data(base64.b64decode(encrypted_age), key)

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

    def save_user_data(self):
        try:
            with open('user_data_directory\\users.json', 'w') as file:
                json.dump(self.user_data, file, indent=4)
                print("User  data saved successfully.")
        except Exception as e:
            print(f"Error saving user data: {e}")

    def create_widgets(self):
        settingslb = tk.Label(self, text="SETTINGS", font=("Arial", 25, "bold"))
        settingslb.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="w")  # Align to the left

        # Load and display user image
        self.display_user_image()  # Call the updated method to load the specific image

        # Display user details

        username = current_user_key  # Use the current user key as the username
        username_label = tk.Label(self, text="Username:", font=("Arial", 14))
        username_label.grid(row=2, column=0, padx=20, pady=5, sticky="e")  # Align label to the right
        username_value = tk.Label(self, text=username, font=("Arial", 14))
        username_value.grid(row=2, column=1, padx=20, pady=5, sticky="w")  # Align value to the left

        email = self.current_user_data.get('email', '')
        email_label = tk.Label(self, text="Email:", font=("Arial", 14))
        email_label.grid(row=3, column=0, padx=20, pady=5, sticky="e")  # Align label to the right
        email_value = tk.Label(self, text=email, font=("Arial", 14))
        email_value.grid(row=3, column=1, padx=20, pady=5, sticky="w")  # Align value to the left

        age = self.current_user_data.get('age', '')
        age_label = tk.Label(self, text="Age:", font=("Arial", 14))
        age_label.grid(row=4, column=0, padx=20, pady=5, sticky="e")  # Align label to the right
        age_value = tk.Label(self, text=age, font=("Arial", 14))
        age_value.grid(row=4, column=1, padx=20, pady=5, sticky="w")  # Align value to the left

        designation = self.current_user_data.get('designation', '')
        designation_label = tk.Label(self, text="Designation:", font=("Arial", 14))
        designation_label.grid(row=5, column=0, padx=20, pady=5, sticky="e")  # Align label to the right
        designation_value = tk.Label(self, text=designation, font=("Arial", 14))
        designation_value.grid(row=5, column=1, padx=20, pady=5, sticky="w")  # Align value to the left

    def display_user_image(self):
        image_path = os.path.join('user_data_directory', 'profile_image.png')  # Specify the path to the image
        if not os.path.exists(image_path):
            print("Image file not found.")
            return

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

