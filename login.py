import customtkinter as ctk
import tkinter.messagebox as tkmb
import os
import json
import re
from cryptography.fernet import Fernet
from PIL import Image
import base64
from utils import decrypt_data, generate_key, MASTER_KEY
import ctypes

# DPI Awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Windows DPI fix
except Exception:
    pass

ctk.set_widget_scaling(1.0)


class LoginApp:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.geometry("1280x720")
        self.app.title("Bloom Sentry")

        self.USER_DATA_FILE = "users.json"
        self.SAVE_DIRECTORY = "user_data_directory"
        self.REMEMBER_ME_FILE = "remember_me.json"
        self.MASTER_KEY = b"YourMasterEncryptionKey"

        self.user_data = self.load_user_data()
        self.remember_me_data = self.load_remember_me()
        self.remember_me = self.remember_me_data.get("remember_me", False)
        self.remembered_username = self.remember_me_data.get("username", "")

        self.current_user_key = None
        self.label = None
        self.logo_image = None
        self.setup_ui()
        self.after_id = None

    def start_dpi_check(self):
        self.after_id = self.app.after(1000, self.check_dpi_scaling)  # Track the ID

    def check_dpi_scaling(self):
        # Your DPI logic here
        self.after_id = self.app.after(1000, self.check_dpi_scaling)  # Reschedule

    def cleanup(self):
        if self.after_id:
            self.app.after_cancel(self.after_id)

    def setup_ui(self):
        logo = Image.open('Icons\\AppLogo.png').resize((300, 100))
        self.logo_image = ctk.CTkImage(light_image=logo, dark_image=logo, size=(300, 100))  # Store the image reference
        self.label = ctk.CTkLabel(self.app, image=self.logo_image, text="")
        self.label.pack(pady=10)

        frame_width = 250
        frame_height = 400
        frame = ctk.CTkFrame(master=self.app, width=frame_width, height=frame_height)
        frame.pack(pady=10, padx=0, anchor='center')
        frame.pack_propagate(False)

        label = ctk.CTkLabel(master=frame, text='LOGIN', font=('Arial', 24))
        label.pack(pady=12, padx=10)

        self.user_entry = ctk.CTkEntry(master=frame, placeholder_text="Username")
        self.user_entry.pack(pady=20, padx=18)

        self.user_pass = ctk.CTkEntry(master=frame, placeholder_text="Password", show="*")
        self.user_pass.pack(pady=20, padx=18)

        self.user_entry.insert(0, self.remembered_username)

        button = ctk.CTkButton(master=frame, text='Login', command=self.login)
        button.pack(pady=20, padx=18)

        self.user_pass.bind("<Return>", lambda event: self.login())

        self.remember_me_checkbox = ctk.CTkCheckBox(master=frame, text='Remember Me')
        self.remember_me_checkbox.pack(pady=20, padx=18)
        self.remember_me_checkbox.select() if self.remember_me else self.remember_me_checkbox.deselect()

        signup_button = ctk.CTkButton(master=frame, text='Sign Up', command=self.signup)
        signup_button.pack(pady=20, padx=18)

        self.center_window(self.app, 1280, 720)
        self.app.mainloop()

    def center_window(self, window, width, height):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        window.geometry(f'{width}x{height}+{x}+{y}')

    def load_user_data(self):
        if not self.create_json_file_if_not_exists(self.USER_DATA_FILE):
            return {}

        try:
            filepath = os.path.join(self.SAVE_DIRECTORY, self.USER_DATA_FILE)
            with open(filepath, "r") as file:
                encrypted_data = file.read()

            if encrypted_data:
                try:
                    user_data_encrypted = json.loads(encrypted_data)
                except json.JSONDecodeError:
                    tkmb.showerror("Error", "User  data file is corrupted.")
                    return {}

                user_data_decrypted = {}
                for username, encrypted_user_data in user_data_encrypted.items():
                    salt = username.encode()
                    key = generate_key(MASTER_KEY.decode(), salt)
                    try:
                        decrypted_password = decrypt_data(base64.b64decode(encrypted_user_data['password']), key)
                        user_data_decrypted[username] = {
                            'password': decrypted_password,
                            'email': decrypt_data(base64.b64decode(encrypted_user_data['email']), key),
                            'designation': decrypt_data(base64.b64decode(encrypted_user_data['designation']), key)
                        }
                    except Exception as e:
                        tkmb.showerror("Error", f"Decryption error for user {username}: {e}")
                        return {}
                return user_data_decrypted
            else:
                return {}
        except FileNotFoundError:
            return {}
        except Exception as e:
            tkmb.showerror(title="Error", message=f"Error loading user data: {e}")
            return {}

    def load_remember_me(self):
        try:
            filepath = os.path.join(self.SAVE_DIRECTORY, self.REMEMBER_ME_FILE)
            with open(filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
        except Exception as e:
            tkmb.showerror("Error", f"Error loading remember me data: {e}")
            return {}

    def create_json_file_if_not_exists(self, filename):
        if not os.path.exists(self.SAVE_DIRECTORY):
            try:
                os.makedirs(self.SAVE_DIRECTORY)
            except OSError as e:
                tkmb.showerror("Error", f"Could not create directory: {e}")
                return False

        filepath = os.path.join(self.SAVE_DIRECTORY, filename)
        isFile = os.path.isfile(filepath)

        if not isFile:
            try:
                with open(filepath, 'w') as fp:
                    json.dump({}, fp)
            except IOError as e:
                tkmb.showerror("Error", f"Could not create file: {e}")
                return False
        return True

    def encrypt_data(self, data, encryption_key):
        f = Fernet(encryption_key)
        encrypted_data = f.encrypt(data.encode())
        return encrypted_data

    def is_valid_username(self, username):
        pattern = r"^[a-zA-Z0-9]{3,}$"
        return bool(re.match(pattern, username))

    def is_valid_password(self, password):
        pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+.,])[A-Za-z\d!@#$%^&*()_+.,]{8,}$"
        return bool(re.match(pattern, password))

    def is_valid_email(self, email):
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    def login(self):
        username = self.user_entry.get()
        password = self.user_pass.get()
        remember = self.remember_me_checkbox.get()

        if not self.is_valid_username(username):
            tkmb.showwarning(title="Invalid Username", message="Username must be alphanumeric and at least 3 chars.")
            return

        if username in self.user_data:
            # Decrypt the stored password
            salt = username.encode()  # Use the username as the salt
            print(salt)
            key = generate_key(MASTER_KEY.decode(), salt)  # Generate the key
            print(key)
            try:
                stored_password = self.user_data.get(username, {}).get('password')
            except Exception as e:
                tkmb.showerror("Error", f"Failed to decrypt password: {e}")
                return

            if password == stored_password:
                current_user_key = username
                self.app.after_cancel("all")
                self.app.destroy()
                from main import Main
                app = Main(current_user_key)
                app.mainloop()
                if remember:
                    self.save_remember_me({"remember_me": True, "username": username})
                else:
                    self.save_remember_me({"remember_me": False, "username": ""})
            else:
                tkmb.showwarning(title='Wrong password', message='Check your password.')
        else:
            tkmb.showerror("Error", "Invalid Username or Password")

    def save_remember_me(self, data):
        try:
            filepath = os.path.join(self.SAVE_DIRECTORY, self.REMEMBER_ME_FILE)
            with open(filepath, "w") as f:
                json.dump(data, f)
        except Exception as e:
            tkmb.showerror("Error", f"Error saving remember me data: {e}")

    def signup(self):
        self.app.withdraw()
        signup_window = ctk.CTkToplevel(self.app)
        signup_window.title("Sign Up")
        signup_window.geometry("1280x720")

        width = 700
        signup_frame = ctk.CTkFrame(master=signup_window, width=width)
        signup_frame.pack(pady=15, padx=40, fill='y', expand=False)

        signup_label = ctk.CTkLabel(master=signup_frame, text='Create an Account')
        signup_label.pack(pady=15, padx=20)

        new_username_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Username")
        new_username_entry.pack(pady=15, padx=20)

        new_password_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Password", show="*")
        new_password_entry.pack(pady=15, padx=20)

        confirm_password_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Confirm Password", show="*")
        confirm_password_entry.pack(pady=15, padx=20)

        email_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Email")
        email_entry.pack(pady=15, padx=20)

        designation_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Designation")
        designation_entry.pack(pady=15, padx=20)

        # Function to handle registration
        def trigger_signup():
            self.register_user(
                new_username_entry.get(),
                new_password_entry.get(),
                confirm_password_entry.get(),
                email_entry.get(),
                designation_entry.get(),
                signup_window
            )

        # Bind Enter key to all entry fields
        entries = [
            new_username_entry,
            new_password_entry,
            confirm_password_entry,
            email_entry,
            designation_entry
        ]
        for entry in entries:
            entry.bind("<Return>", lambda event: trigger_signup())

        # Signup button
        signup_button = ctk.CTkButton(
            master=signup_frame,
            text='Sign Up',
            command=trigger_signup  # Use the same function
        )
        signup_button.pack(pady=30, padx=10)

        login_instead_label = ctk.CTkLabel(master=signup_frame, text="Already have an account?")
        login_instead_label.pack(pady=10, padx=10)

        login_instead_button = ctk.CTkButton(
            master=signup_frame,
            text="Login",
            command=lambda: [signup_window.destroy(), self.app.deiconify()]
        )
        login_instead_button.pack(pady=30, padx=15)

        self.center_window(signup_window, 1280, 720)
        signup_window.protocol("WM_DELETE_WINDOW", lambda: (self.app.deiconify(), signup_window.destroy()))

    def register_user(self, new_username, new_password, confirm_password, email, designation, signup_window):
        if not self.is_valid_username(new_username):
            tkmb.showwarning(title="Invalid Username", message="Alphanumeric, min 3 chars.")
            return

        if not self.is_valid_password(new_password):
            tkmb.showwarning(title="Invalid Password", message="Min 8 chars, uppercase, lowercase, digit, special.")
            return

        if not self.is_valid_email(email):
            tkmb.showwarning(title="Invalid Email", message="Please enter a valid email address.")
            return

        if not designation:
            tkmb.showwarning(title="Invalid Designation", message="Designation cannot be empty.")
            return

        if new_username in self.user_data:
            tkmb.showerror(title="Signup Failed", message="Username exists.")
            return

        if new_password != confirm_password:
            tkmb.showerror(title="Signup Failed", message="Passwords do not match.")
            return

        self.user_data[new_username] = {
            'password': new_password,
            'email': email,
            'designation': designation
        }

        if self.save_user_data(self.user_data):
            tkmb.showinfo(title="Signup Successful", message="Account created!")
            signup_window.destroy()
            self.app.deiconify()
        else:
            tkmb.showerror("Error", "Signup failed.")

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


if __name__ == "__main__":
    LoginApp()
