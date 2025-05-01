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

# Set appearance mode and default color theme
ctk.set_appearance_mode("Light")  # Light mode to match the design
ctk.set_default_color_theme("blue")
ctk.set_widget_scaling(1.0)


class LoginApp:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.geometry("1000x600")
        self.app.title("Bloom Sentry")
        self.app.iconbitmap("Icons/favicon.ico") if os.path.exists("Icons/favicon.ico") else None

        # Configure grid layout
        self.app.grid_columnconfigure(0, weight=4)  # Blue section (40% of width)
        self.app.grid_columnconfigure(1, weight=6)  # White section (60% of width)
        self.app.grid_rowconfigure(0, weight=1)

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
        self.wave_image = None

        # Check if there's already a master user
        self.has_master_user = self.check_master_user_exists()

        self.setup_ui()
        self.after_id = None
        self.start_dpi_check()

    def check_master_user_exists(self):
        """Check if a master user already exists in the system"""
        for username, data in self.user_data.items():
            user_type = data.get('user_type', 'regular')
            if user_type == 'master':
                return True
        return False

    def start_dpi_check(self):
        self.after_id = self.app.after(1000, self.check_dpi_scaling)

    def check_dpi_scaling(self):
        # DPI scaling logic here
        self.after_id = self.app.after(1000, self.check_dpi_scaling)

    def cleanup(self):
        if self.after_id:
            self.app.after_cancel(self.after_id)

    def setup_ui(self):
        # Create blue section (left side with logo)
        blue_frame = ctk.CTkFrame(self.app, corner_radius=0)
        blue_frame.grid(row=0, column=0, sticky="nsew")

        # Set the blue color to a lighter shade - CHANGED FROM #2B8CD8 to #65B4FF
        blue_frame.configure(fg_color="#65B4FF")  # Lighter blue color

        # Load and center the logo
        logo_frame = ctk.CTkFrame(blue_frame, fg_color="transparent")
        logo_frame.place(relx=0.5, rely=0.45, anchor="center")

        try:
            logo = Image.open('Icons/AppLogo.png')
            logo_width, logo_height = logo.size
            # Maintain aspect ratio
            display_width = 180
            display_height = int(logo_height * (display_width / logo_width))
            self.logo_image = ctk.CTkImage(light_image=logo, dark_image=logo, size=(display_width, display_height))
            logo_label = ctk.CTkLabel(logo_frame, image=self.logo_image, text="", text_color="white")
            logo_label.pack()
        except:
            # If logo not found, create text and styled placeholder based on the image
            # Blue triangle with green curve
            logo_canvas = ctk.CTkCanvas(logo_frame, width=180, height=100,
                                        bg=blue_frame.cget("fg_color"), highlightthickness=0)
            logo_canvas.pack()

            # Blue triangle (droplet) - UPDATED COLOR
            logo_canvas.create_polygon(90, 20, 40, 80, 140, 80, fill="#65B4FF", outline="#54A3EE", width=2)

            # Green curve
            logo_canvas.create_arc(30, 60, 150, 100, start=0, extent=-180, fill="#92D050", outline="#92D050")

            # Add "Bloom Sentry" text with WHITE color
            logo_text = ctk.CTkLabel(logo_frame, text="Bloom Sentry",
                                     font=ctk.CTkFont(family="Arial", size=24, weight="bold"),
                                     text_color="#FFFFFF")
            logo_text.pack(pady=(10, 0))

        # Copyright text at bottom
        copyright_label = ctk.CTkLabel(blue_frame, text="© 2025 Terra. All rights reserved.",
                                       font=ctk.CTkFont(size=10),
                                       text_color="#FFFFFF")
        copyright_label.place(relx=0.5, rely=0.95, anchor="center")

        # Copyright text at bottom
        copyright_label = ctk.CTkLabel(blue_frame, text="© 2025 Terra. All rights reserved.",
                                       font=ctk.CTkFont(size=10),
                                       text_color="#FFFFFF")
        copyright_label.place(relx=0.5, rely=0.95, anchor="center")

        # Create white section (right side with login form)
        white_frame = ctk.CTkFrame(self.app, fg_color="#FFFFFF", corner_radius=0)
        white_frame.grid(row=0, column=1, sticky="nsew")

        # Login form container
        form_frame = ctk.CTkFrame(white_frame, fg_color="transparent")
        form_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.7, relheight=0.8)

        # Welcome text
        welcome_label = ctk.CTkLabel(form_frame, text="Welcome",
                                     font=ctk.CTkFont(family="Arial", size=24, weight="bold"),
                                     text_color="#333333")
        welcome_label.pack(anchor="w", pady=(0, 5))

        welcome_text = ctk.CTkLabel(form_frame, text="Login to your account to continue",
                                    font=ctk.CTkFont(size=12),
                                    text_color="#888888")
        welcome_text.pack(anchor="w", pady=(0, 30))

        # Username field
        username_label = ctk.CTkLabel(form_frame, text="Email",
                                      font=ctk.CTkFont(size=12),
                                      text_color="#333333")
        username_label.pack(anchor="w", pady=(0, 5))

        self.user_entry = ctk.CTkEntry(master=form_frame, placeholder_text="Your email or username",
                                       height=40, corner_radius=8, border_width=1,
                                       fg_color="#F1F5F9", border_color="#E2E8F0")
        self.user_entry.pack(fill="x", pady=(0, 15))
        self.user_entry.insert(0, self.remembered_username)

        # Password field
        password_label = ctk.CTkLabel(form_frame, text="Password",
                                      font=ctk.CTkFont(size=12),
                                      text_color="#333333")
        password_label.pack(anchor="w", pady=(0, 5))

        self.user_pass = ctk.CTkEntry(master=form_frame, placeholder_text="Your password",
                                      height=40, corner_radius=8, border_width=1, show="•",
                                      fg_color="#F1F5F9", border_color="#E2E8F0")
        self.user_pass.pack(fill="x")
        self.user_pass.bind("<Return>", lambda event: self.login())

        # Forgot password and remember me in one row
        options_frame = ctk.CTkFrame(master=form_frame, fg_color="transparent")
        options_frame.pack(fill="x", pady=(10, 30))

        self.remember_me_checkbox = ctk.CTkCheckBox(master=options_frame, text='Remember me',
                                                    font=ctk.CTkFont(size=12),
                                                    checkbox_height=16, checkbox_width=16,
                                                    corner_radius=4)
        self.remember_me_checkbox.pack(side="left")
        self.remember_me_checkbox.select() if self.remember_me else self.remember_me_checkbox.deselect()

        forgot_pass_btn = ctk.CTkButton(master=options_frame, text='Forgot your password?',
                                        font=ctk.CTkFont(size=12),
                                        fg_color="transparent", text_color="#3B8ED0",
                                        hover=False, height=20, command=self.forgot_password)
        forgot_pass_btn.pack(side="right")

        # Login button
        login_button = ctk.CTkButton(master=form_frame, text='Log in',
                                     font=ctk.CTkFont(size=14),
                                     corner_radius=8, height=40,
                                     fg_color="#4CAF50", hover_color="#388E3C",
                                     command=self.login)
        login_button.pack(fill="x", pady=(0, 20))

        # Sign up option
        signup_frame = ctk.CTkFrame(master=form_frame, fg_color="transparent")
        signup_frame.pack()

        no_account_label = ctk.CTkLabel(master=signup_frame, text="Don't have an account?",
                                        font=ctk.CTkFont(size=12),
                                        text_color="#888888")
        no_account_label.pack(side="left", padx=(0, 5))

        signup_btn = ctk.CTkButton(master=signup_frame, text='Sign up',
                                   font=ctk.CTkFont(size=12),
                                   fg_color="transparent", text_color="#3B8ED0",
                                   hover=False, width=50, height=20, command=self.signup)
        signup_btn.pack(side="left")

        self.center_window(self.app, 1000, 600)
        self.app.mainloop()

    def change_appearance_mode(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)

    def forgot_password(self):
        # Placeholder for password recovery functionality
        tkmb.showinfo("Password Recovery", "Password recovery feature will be available soon.")

    def center_window(self, window, width, height):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        window.geometry(f'{width}x{height}+{x}+{y}')

    # Load user data from encrypted file
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
        auth_window.geometry("450x350")
        auth_window.grab_set()  # Make this window modal

        # Create a modern authentication dialog
        frame = ctk.CTkFrame(master=auth_window, corner_radius=15)
        frame.pack(pady=20, padx=20, fill='both', expand=True)

        # Header
        header_frame = ctk.CTkFrame(master=frame, fg_color="transparent")
        header_frame.pack(pady=(20, 10), padx=10)

        # Try to use the logo here too for consistency
        try:
            small_logo = Image.open('Icons/AppLogo.png')
            logo_width, logo_height = small_logo.size
            display_width = 100
            display_height = int(logo_height * (display_width / logo_width))
            small_logo_image = ctk.CTkImage(light_image=small_logo, dark_image=small_logo,
                                            size=(display_width, display_height))
            logo_label = ctk.CTkLabel(header_frame, image=small_logo_image, text="")
            logo_label.pack(pady=(0, 10))
        except:
            pass

        label = ctk.CTkLabel(master=header_frame, text="Master Authentication",
                             font=ctk.CTkFont(family="Arial", size=18, weight="bold"))
        label.pack()

        info_label = ctk.CTkLabel(master=frame,
                                  text="Creating a superuser requires master approval.\nPlease enter master credentials.",
                                  font=ctk.CTkFont(family="Arial", size=12))
        info_label.pack(pady=(5, 20))

        # Username field
        username_label = ctk.CTkLabel(frame, text="Master Username",
                                      font=ctk.CTkFont(size=12),
                                      text_color="#333333")
        username_label.pack(anchor="w", padx=30, pady=(0, 5))

        username_entry = ctk.CTkEntry(master=frame, placeholder_text="Master Username",
                                      height=40, corner_radius=8, border_width=1,
                                      fg_color="#F1F5F9", border_color="#E2E8F0")
        username_entry.pack(fill="x", padx=30, pady=(0, 15))

        # Password field
        password_label = ctk.CTkLabel(frame, text="Master Password",
                                      font=ctk.CTkFont(size=12),
                                      text_color="#333333")
        password_label.pack(anchor="w", padx=30, pady=(0, 5))

        password_entry = ctk.CTkEntry(master=frame, placeholder_text="Master Password", show="•",
                                      height=40, corner_radius=8, border_width=1,
                                      fg_color="#F1F5F9", border_color="#E2E8F0")
        password_entry.pack(fill="x", padx=30, pady=(0, 25))

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

        # Button frame
        button_frame = ctk.CTkFrame(master=frame, fg_color="transparent")
        button_frame.pack(pady=10, padx=30, fill="x")

        # Updated button color to match new blue
        auth_button = ctk.CTkButton(master=button_frame, text="Authenticate",
                                    font=ctk.CTkFont(family="Arial", size=14),
                                    corner_radius=8, height=40,
                                    fg_color="#65B4FF", hover_color="#54A3EE",
                                    command=verify_master)
        auth_button.pack(side="left", padx=(0, 10), fill="x", expand=True)

        cancel_button = ctk.CTkButton(master=button_frame, text="Cancel",
                                      font=ctk.CTkFont(family="Arial", size=14),
                                      corner_radius=8, height=40,
                                      fg_color="#E74C3C", hover_color="#C0392B",
                                      command=lambda: auth_window.destroy())
        cancel_button.pack(side="left", fill="x", expand=True)

        # Center the authentication window
        self.center_window(auth_window, 450, 350)

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
        signup_window.geometry("1000x600")

        # Split layout for signup
        signup_window.grid_columnconfigure(0, weight=4)  # Blue panel side (40%)
        signup_window.grid_columnconfigure(1, weight=6)  # Form side (60%)
        signup_window.grid_rowconfigure(0, weight=1)

        # Blue section (left side) - UPDATED COLOR
        blue_frame = ctk.CTkFrame(signup_window, corner_radius=0)
        blue_frame.grid(row=0, column=0, sticky="nsew")
        blue_frame.configure(fg_color="#65B4FF")  # Lighter blue color

        # Add logo in center
        logo_frame = ctk.CTkFrame(blue_frame, fg_color="transparent")
        logo_frame.place(relx=0.5, rely=0.45, anchor="center")

        try:
            logo_label = ctk.CTkLabel(logo_frame, image=self.logo_image, text="")
            logo_label.pack()
        except:
            # If logo not loaded in login, recreate it here
            try:
                logo = Image.open('Icons/AppLogo.png')
                logo_width, logo_height = logo.size
                display_width = 180
                display_height = int(logo_height * (display_width / logo_width))
                self.logo_image = ctk.CTkImage(light_image=logo, dark_image=logo, size=(display_width, display_height))
                logo_label = ctk.CTkLabel(logo_frame, image=self.logo_image, text="")
                logo_label.pack()
            except:
                # If logo still not found, create placehoder
                logo_canvas = ctk.CTkCanvas(logo_frame, width=180, height=100,
                                            bg=blue_frame.cget("fg_color"), highlightthickness=0)
                logo_canvas.pack()

                # Blue triangle (droplet) - UPDATED COLOR
                logo_canvas.create_polygon(90, 20, 40, 80, 140, 80, fill="#65B4FF", outline="#54A3EE", width=2)

                # Green curve
                logo_canvas.create_arc(30, 60, 150, 100, start=0, extent=-180, fill="#92D050", outline="#92D050")

                # Add "Bloom Sentry" text in WHITE (not blue)
                logo_text = ctk.CTkLabel(logo_frame, text="Bloom Sentry",
                                         font=ctk.CTkFont(family="Arial", size=24, weight="bold"),
                                         text_color="#FFFFFF")
                logo_text.pack(pady=(10, 0))

        # Info text
        info_label = ctk.CTkLabel(blue_frame, text="Create your account",
                                  font=ctk.CTkFont(family="Arial", size=16),
                                  text_color="#FFFFFF")
        info_label.place(relx=0.5, rely=0.57, anchor="center")

        # Copyright text at bottom
        copyright_label = ctk.CTkLabel(blue_frame, text="© 2025 Terra. All rights reserved.",
                                       font=ctk.CTkFont(size=10),
                                       text_color="#FFFFFF")
        copyright_label.place(relx=0.5, rely=0.95, anchor="center")

        # Copyright text
        copyright_label = ctk.CTkLabel(blue_frame, text="© 2025 Terra. All rights reserved.",
                                       font=ctk.CTkFont(size=10),
                                       text_color="#FFFFFF")
        copyright_label.place(relx=0.5, rely=0.95, anchor="center")

        # White section (right side with signup form)
        white_frame = ctk.CTkFrame(signup_window, fg_color="#FFFFFF", corner_radius=0)
        white_frame.grid(row=0, column=1, sticky="nsew")

        # Signup form container
        form_frame = ctk.CTkFrame(white_frame, fg_color="transparent")
        form_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.7, relheight=0.85)

        # Title
        title_label = ctk.CTkLabel(form_frame, text="Sign Up",
                                   font=ctk.CTkFont(family="Arial", size=24, weight="bold"),
                                   text_color="#333333")
        title_label.pack(anchor="w", pady=(0, 5))

        subtitle = ctk.CTkLabel(form_frame, text="Please fill in the form to create your account",
                                font=ctk.CTkFont(size=12),
                                text_color="#888888")
        subtitle.pack(anchor="w", pady=(0, 20))

        # Create scrollable frame for the form
        form_container = ctk.CTkScrollableFrame(form_frame, fg_color="transparent",
                                                scrollbar_button_color="#65B4FF",
                                                scrollbar_button_hover_color="#54A3EE")
        form_container.pack(fill="both", expand=True)

        # Username field
        username_label = ctk.CTkLabel(form_container, text="Username",
                                      font=ctk.CTkFont(size=12),
                                      text_color="#333333")
        username_label.pack(anchor="w", pady=(0, 5))

        new_username_entry = ctk.CTkEntry(master=form_container, placeholder_text="At least 3 alphanumeric characters",
                                          height=40, corner_radius=8, border_width=1,
                                          fg_color="#F1F5F9", border_color="#E2E8F0")
        new_username_entry.pack(fill="x", pady=(0, 15))

        # Password field
        password_label = ctk.CTkLabel(form_container, text="Password",
                                      font=ctk.CTkFont(size=12),
                                      text_color="#333333")
        password_label.pack(anchor="w", pady=(0, 5))

        new_password_entry = ctk.CTkEntry(master=form_container,
                                          placeholder_text="Min 8 chars with upper, lower, digit & symbol",
                                          height=40, corner_radius=8, border_width=1, show="•",
                                          fg_color="#F1F5F9", border_color="#E2E8F0")
        new_password_entry.pack(fill="x", pady=(0, 5))

        # Password strength indicator
        password_strength_frame = ctk.CTkFrame(master=form_container, height=4, fg_color="#E0E0E0")
        password_strength_frame.pack(fill="x", pady=(0, 5))

        password_strength_label = ctk.CTkLabel(form_container, text="Password Strength: None",
                                               font=ctk.CTkFont(size=10), text_color="#888888")
        password_strength_label.pack(anchor="e", pady=(0, 15))

        # Function to check password strength in real-time
        def check_password_strength(event=None):
            password = new_password_entry.get()
            strength = 0
            feedback = "None"
            color = "#E0E0E0"  # Default gray

            if len(password) >= 8:
                strength += 1
            if re.search(r"[A-Z]", password):
                strength += 1
            if re.search(r"[a-z]", password):
                strength += 1
            if re.search(r"[0-9]", password):
                strength += 1
            if re.search(r"[!@#$%^&*()_+.,]", password):
                strength += 1

            if strength == 0:
                feedback = "None"
                color = "#E0E0E0"
            elif strength <= 2:
                feedback = "Weak"
                color = "#FF6B6B"
            elif strength <= 4:
                feedback = "Medium"
                color = "#FFD166"
            else:
                feedback = "Strong"
                color = "#06D6A0"

            password_strength_frame.configure(fg_color=color)
            password_strength_label.configure(text=f"Password Strength: {feedback}")

        new_password_entry.bind("<KeyRelease>", check_password_strength)

        # Confirm Password
        confirm_password_label = ctk.CTkLabel(form_container, text="Confirm Password",
                                              font=ctk.CTkFont(size=12),
                                              text_color="#333333")
        confirm_password_label.pack(anchor="w", pady=(0, 5))

        confirm_password_entry = ctk.CTkEntry(master=form_container, placeholder_text="Confirm your password",
                                              height=40, corner_radius=8, border_width=1, show="•",
                                              fg_color="#F1F5F9", border_color="#E2E8F0")
        confirm_password_entry.pack(fill="x", pady=(0, 15))

        # Designation field
        designation_label = ctk.CTkLabel(form_container, text="Designation",
                                         font=ctk.CTkFont(size=12),
                                         text_color="#333333")
        designation_label.pack(anchor="w", pady=(0, 5))

        designation_entry = ctk.CTkEntry(master=form_container, placeholder_text="Your role or position",
                                         height=40, corner_radius=8, border_width=1,
                                         fg_color="#F1F5F9", border_color="#E2E8F0")
        designation_entry.pack(fill="x", pady=(0, 15))

        user_type_var = ctk.StringVar(value="regular")

        # Terms and conditions checkbox
        terms_checkbox = ctk.CTkCheckBox(master=form_container, text="I agree to the Terms & Conditions",
                                         font=ctk.CTkFont(size=12),
                                         checkbox_height=16, checkbox_width=16,
                                         corner_radius=4)
        terms_checkbox.pack(anchor="w", pady=(0, 20))

        # Buttons frame at bottom of form container
        button_frame = ctk.CTkFrame(form_frame, fg_color="transparent", height=50)
        button_frame.pack(fill="x", pady=(10, 0))

        # Function to handle registration
        def trigger_signup():
            # First check terms agreement
            if not terms_checkbox.get():
                tkmb.showwarning("Terms Required", "You must agree to the Terms & Conditions to continue.")
                return

            self.register_user(
                new_username_entry.get(),
                new_password_entry.get(),
                confirm_password_entry.get(),
                designation_entry.get(),  # Using designation only
                signup_window
            )

        # Signup button
        signup_button = ctk.CTkButton(
            master=button_frame,
            text='SIGN UP',
            font=ctk.CTkFont(size=14),
            corner_radius=8,
            height=40,
            fg_color="#4CAF50",
            hover_color="#388E3C",
            command=trigger_signup
        )
        signup_button.pack(side="left", padx=(0, 10), fill="x", expand=True)

        # Cancel button
        cancel_button = ctk.CTkButton(
            master=button_frame,
            text="CANCEL",
            font=ctk.CTkFont(size=14),
            corner_radius=8,
            height=40,
            fg_color="#E74C3C",
            hover_color="#C0392B",
            command=lambda: [signup_window.destroy(), self.app.deiconify()]
        )
        cancel_button.pack(side="left", fill="x", expand=True)

        # Bind Enter key to all entry fields
        entries = [
            new_username_entry,
            new_password_entry,
            confirm_password_entry,
            designation_entry
        ]
        for entry in entries:
            entry.bind("<Return>", lambda event: trigger_signup())

        # Already have an account text at bottom
        login_link_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        login_link_frame.pack(pady=(15, 0))

        login_instead_label = ctk.CTkLabel(master=login_link_frame, text="Already have an account?",
                                           font=ctk.CTkFont(size=12),
                                           text_color="#888888")
        login_instead_label.pack(side="left", padx=(0, 5))

        login_link = ctk.CTkButton(
            master=login_link_frame,
            text="Login here",
            font=ctk.CTkFont(size=12),
            fg_color="transparent",
            text_color="#3B8ED0",
            hover=False,
            width=50,
            height=20,
            command=lambda: [signup_window.destroy(), self.app.deiconify()]
        )
        login_link.pack(side="left")

        self.center_window(signup_window, 1000, 600)
        signup_window.protocol("WM_DELETE_WINDOW", lambda: (self.app.deiconify(), signup_window.destroy()))

    def register_user(self, new_username, new_password, confirm_password, designation, signup_window):
        if not self.is_valid_username(new_username):
            tkmb.showwarning(title="Invalid Username",
                             message="Username must be alphanumeric and at least 3 characters.")
            return

        if not self.is_valid_password(new_password):
            tkmb.showwarning(title="Invalid Password",
                             message="Password must have at least 8 characters, include uppercase, lowercase, a digit, and a special character.")
            return

        if not designation:
            tkmb.showwarning(title="Invalid Designation", message="Designation cannot be empty.")
            return

        if new_username in self.user_data:
            tkmb.showerror(title="Signup Failed", message="Username already exists.")
            return

        if new_password != confirm_password:
            tkmb.showerror(title="Signup Failed", message="Passwords do not match.")
            return

        # Always set user_type to 'regular' upon signup
        user_type = "regular"

        # Create the new user
        self.user_data[new_username] = {
            'password': new_password,
            'designation': designation,
            'user_type': user_type
        }

        if self.save_user_data(self.user_data):
            tkmb.showinfo(title="Signup Successful", message="Account created successfully!")
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
            tkmb.showerror(title="Error", message=f"Error saving user data: {e}")
            return False


if __name__ == "__main__":
    LoginApp()
