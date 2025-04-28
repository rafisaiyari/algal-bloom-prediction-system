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

        # Check if there's already a master user
        self.has_master_user = self.check_master_user_exists()

        self.setup_ui()
        self.after_id = None

    def check_master_user_exists(self):
        """Check if a master user already exists in the system"""
        for username, data in self.user_data.items():
            user_type = data.get('user_type', 'regular')
            if user_type == 'master':
                return True
        return False

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
        self.remember_me_checkbox.pack(pady=10, padx=18)
        self.remember_me_checkbox.select() if self.remember_me else self.remember_me_checkbox.deselect()

        signup_button = ctk.CTkButton(master=frame, text='Sign Up', command=self.signup)
        signup_button.pack(pady=10, padx=18)

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
                    tkmb.showerror("Error", "User data file is corrupted.")
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

                        # Add user type to the decrypted data
                        if 'user_type' in encrypted_user_data:
                            user_data_decrypted[username]['user_type'] = decrypt_data(
                                base64.b64decode(encrypted_user_data['user_type']),
                                key
                            )
                        else:
                            user_data_decrypted[username]['user_type'] = 'regular'  # Default to regular user

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
            key = generate_key(MASTER_KEY.decode(), salt)  # Generate the key

            try:
                stored_password = self.user_data.get(username, {}).get('password')
                user_type = self.user_data[username].get('user_type', 'regular')
            except Exception as e:
                tkmb.showerror("Error", f"Failed to decrypt password: {e}")
                return

            # Password validation
            if password != stored_password:
                tkmb.showwarning(title='Wrong password', message='Check your password.')
                return

            # Successfully authenticated
            current_user_key = username
            self.app.after_cancel("all")
            self.app.destroy()

            # Launch main application with user's registered type
            from main import Main
            app = Main(current_user_key, user_type)
            app.mainloop()

            if remember:
                self.save_remember_me({"remember_me": True, "username": username})
            else:
                self.save_remember_me({"remember_me": False, "username": ""})
        else:
            tkmb.showerror("Error", "Invalid Username or Password")

    def authenticate_master_user(self, parent_window):
        """Show a popup for master user authentication and return True if successful"""
        master_auth_result = [False]  # Use a list to store the result (mutable)

        auth_window = ctk.CTkToplevel(parent_window)
        auth_window.title("Master User Authentication")
        auth_window.geometry("400x300")
        auth_window.grab_set()  # Make this window modal

        frame = ctk.CTkFrame(master=auth_window)
        frame.pack(pady=20, padx=20, fill='both', expand=True)

        label = ctk.CTkLabel(master=frame, text="Master User Authentication Required", font=('Arial', 16))
        label.pack(pady=10, padx=10)

        info_label = ctk.CTkLabel(master=frame, text="Creating a superuser requires master approval",
                                  font=('Arial', 12))
        info_label.pack(pady=5, padx=10)

        username_entry = ctk.CTkEntry(master=frame, placeholder_text="Master Username")
        username_entry.pack(pady=10, padx=20)

        password_entry = ctk.CTkEntry(master=frame, placeholder_text="Master Password", show="*")
        password_entry.pack(pady=10, padx=20)

        def verify_master():
            master_username = username_entry.get()
            master_password = password_entry.get()

            # Check if the user exists and is a master
            if master_username in self.user_data:
                user_data = self.user_data[master_username]
                user_type = user_data.get('user_type', 'regular')

                if user_type == 'master' and user_data['password'] == master_password:
                    master_auth_result[0] = True
                    auth_window.destroy()
                    return

            tkmb.showerror("Authentication Failed", "Invalid master credentials", parent=auth_window)

        auth_button = ctk.CTkButton(master=frame, text="Authenticate", command=verify_master)
        auth_button.pack(pady=20, padx=20)

        cancel_button = ctk.CTkButton(master=frame, text="Cancel",
                                      command=lambda: auth_window.destroy())
        cancel_button.pack(pady=10, padx=20)

        # Center the authentication window
        self.center_window(auth_window, 400, 300)

        # Wait until this window is destroyed
        parent_window.wait_window(auth_window)

        return master_auth_result[0]

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

        # Add user type selection
        user_type_frame = ctk.CTkFrame(master=signup_frame)
        user_type_frame.pack(pady=10, padx=20)

        user_type_label = ctk.CTkLabel(master=user_type_frame, text="User Type:")
        user_type_label.pack(side="left", padx=5)

        user_type_var = ctk.StringVar(value="regular")

        # Determine available user types for signup
        if self.has_master_user:
            user_types = ["regular", "superuser"]  # Regular and superuser options when master exists
        else:
            user_types = ["regular", "master"]  # First user can be a master

        user_type_menu = ctk.CTkOptionMenu(
            master=user_type_frame,
            values=user_types,
            variable=user_type_var
        )
        user_type_menu.pack(side="left", padx=5)

        # Function to handle registration
        def trigger_signup():
            user_type = user_type_var.get()

            # If superuser selected, verify with master user first
            if user_type == "superuser":
                if not self.authenticate_master_user(signup_window):
                    # Master authentication failed or was cancelled
                    return

            self.register_user(
                new_username_entry.get(),
                new_password_entry.get(),
                confirm_password_entry.get(),
                email_entry.get(),
                designation_entry.get(),
                user_type,
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
            command=trigger_signup
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

    def register_user(self, new_username, new_password, confirm_password, email, designation, user_type, signup_window):
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

        # Special validation for master user
        if user_type == "master" and self.has_master_user:
            tkmb.showerror(title="Signup Failed", message="A master user already exists.")
            return

        # Create the new user
        self.user_data[new_username] = {
            'password': new_password,
            'email': email,
            'designation': designation,
            'user_type': user_type
        }

        if self.save_user_data(self.user_data):
            # If this was the first master user, update our state
            if user_type == "master":
                self.has_master_user = True

            tkmb.showinfo(title="Signup Successful", message=f"Account created as {user_type} user!")
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


if __name__ == "__main__":
    LoginApp()
