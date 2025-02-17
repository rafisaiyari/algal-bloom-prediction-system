import customtkinter as ctk
import tkinter.messagebox as tkmb
import os
import json
import re  # For regular expression-based password validation
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from PIL import Image
import base64

global signup_window
logo = Image.open('AppLogo.png')

# Selecting GUI theme - dark, light, system (for system default)
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

def center_window(window, width, height):
    """Centers a tkinter or ctk window.

    Args:
        window: The tkinter/ctk window to center.
        width: The width of the window.
        height: The height of the window.
    """
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))
    window.geometry(f'{width}x{height}+{x}+{y}')


app = ctk.CTk()
app.geometry("1280x720")
app.title("Algalator")

USER_DATA_FILE = "users.json"
SAVE_DIRECTORY = "user_data_directory"  # MUST NOT be hardcoded in real apps

REMEMBER_ME_FILE = "remember_me.json"

# Master Encryption Key.
MASTER_KEY = b"YourMasterEncryptionKey"  # Replace with a strong, randomly generated key

def generate_key(password, salt):
    """
    Generates a Fernet encryption key from a password and salt using PBKDF2HMAC.
    Args:
        password (str): The password to derive the key from.
        salt (bytes): A randomly generated salt.
    Returns:
        bytes: A 32-byte encryption key.
    """
    password_encoded = password.encode()  # Encode the password to bytes
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password_encoded))
    return key

def encrypt_data(data, encryption_key):
    """Encrypts data using Fernet encryption.
    Args:
        data (str): The data to encrypt.
        encryption_key (bytes): The encryption key.
    Returns:
        bytes: The encrypted data.
    """
    f = Fernet(encryption_key)
    encrypted_data = f.encrypt(data.encode())
    return encrypted_data

def decrypt_data(encrypted_data, encryption_key):
    """Decrypts data using Fernet decryption.
    Args:
        encrypted_data (bytes): The encrypted data.
        encryption_key (bytes): The encryption key.
    Returns:
        str: The decrypted data.
    """
    f = Fernet(encryption_key)
    decrypted_data = f.decrypt(encrypted_data).decode()
    return decrypted_data

def create_json_file_if_not_exists(filename):
    """Creates an empty JSON file if one does not exist."""
    if not os.path.exists(SAVE_DIRECTORY):
        try:
            os.makedirs(SAVE_DIRECTORY)
        except OSError as e:
            tkmb.showerror("Error", f"Could not create directory: {e}")
            return False

    filepath = os.path.join(SAVE_DIRECTORY, filename)
    isFile = os.path.isfile(filepath)

    if not isFile:
        try:
            with open(filepath, 'w') as fp:
                json.dump({}, fp)
        except IOError as e:
            tkmb.showerror("Error", f"Could not create file: {e}")
            return False
    return True

def load_remember_me():
    """Loads 'remember me' data from a JSON file."""
    try:
        filepath = os.path.join(SAVE_DIRECTORY, REMEMBER_ME_FILE)
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    except Exception as e:
        tkmb.showerror("Error", f"Error loading remember me data: {e}")
        return {}

def save_remember_me(data):
    """Saves 'remember me' data to a JSON file."""
    try:
        filepath = os.path.join(SAVE_DIRECTORY, REMEMBER_ME_FILE)
        with open(filepath, "w") as f:
            json.dump(data, f)
    except Exception as e:
        tkmb.showerror("Error", f"Error saving remember me data: {e}")

def load_user_data():
    """Loads user data from the encrypted JSON file."""
    if not create_json_file_if_not_exists(USER_DATA_FILE):
        return {}

    try:
        filepath = os.path.join(SAVE_DIRECTORY, USER_DATA_FILE)
        with open(filepath, "r") as file:
            encrypted_data = file.read()

        if encrypted_data:
            try:  # Try to parse the JSON.  If it fails, it's not valid JSON.
                user_data_encrypted = json.loads(encrypted_data)
            except json.JSONDecodeError:
                tkmb.showerror("Error", "User data file is corrupted.")
                return {}

            user_data_decrypted = {}
            for username, encrypted_user_data in user_data_encrypted.items():
                salt = username.encode()
                key = generate_key(MASTER_KEY.decode(), salt)
                try:
                    decrypted_password = decrypt_data(base64.b64decode(encrypted_user_data['password']), key)  # Adjusted for user data dict
                    user_data_decrypted[username] = {
                        'password': decrypted_password,
                        'age': decrypt_data(base64.b64decode(encrypted_user_data['age']), key), # Decrypt the age
                        'designation': decrypt_data(base64.b64decode(encrypted_user_data['designation']), key) # Decrypt the designation
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

def save_user_data(data):
    """Saves user data to the encrypted JSON file."""
    try:
        user_data_encrypted = {}
        for username, user_data in data.items(): # Changed to iterate through the data
            salt = username.encode()
            key = generate_key(MASTER_KEY.decode(), salt)

            encrypted_password = encrypt_data(user_data['password'], key) # Encrypt password
            encrypted_password_b64 = base64.b64encode(encrypted_password).decode()

            encrypted_age = encrypt_data(user_data['age'], key) # Encrypt age
            encrypted_age_b64 = base64.b64encode(encrypted_age).decode()

            encrypted_designation = encrypt_data(user_data['designation'], key) # Encrypt designation
            encrypted_designation_b64 = base64.b64encode(encrypted_designation).decode()


            user_data_encrypted[username] = {
                'password': encrypted_password_b64,
                'age': encrypted_age_b64,
                'designation': encrypted_designation_b64
            }

        json_string = json.dumps(user_data_encrypted)

        filepath = os.path.join(SAVE_DIRECTORY, USER_DATA_FILE)
        with open(filepath, "w") as file:
            file.write(json_string)
        return True
    except Exception as e:
        tkmb.showerror(title="Error", message=f"Error saving user data: {e}")
        return False

# Load user data at the start
user_data = load_user_data()

# Load remember me data
remember_me_data = load_remember_me()
remember_me = remember_me_data.get("remember_me", False)
remembered_username = remember_me_data.get("username", "")

def is_valid_username(username):
    """Validates the username."""
    pattern = r"^[a-zA-Z0-9]{3,}$"  # Alphanumeric, min 3 chars
    return bool(re.match(pattern, username))

def is_valid_password(password):
    """Validates the password."""
    # Ensure password contains at least one uppercase, one lowercase, one digit, and one special character
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+.,])[A-Za-z\d!@#$%^&*()_+.,]{8,}$"
    return bool(re.match(pattern, password))

def login():
    username = user_entry.get()
    password = user_pass.get()
    remember = remember_me_checkbox.get()

    if not is_valid_username(username):
        tkmb.showwarning(title="Invalid Username", message="Username must be alphanumeric and at least 3 chars.")
        return

    if username in user_data:
        if 'password' in user_data[username]:
            stored_password = user_data[username]['password']
            if password == stored_password:
                tkmb.showinfo(title="Login Successful", message="Logged in!")

                # Save "remember me" data
                if remember:
                    save_remember_me({"remember_me": True, "username": username})
                else:
                    save_remember_me({"remember_me": False, "username": ""})
            else:
                tkmb.showwarning(title='Wrong password', message='Check your password.')
        else:
            tkmb.showerror(title="Login Failed", message="Password not found for this user.")
    else:
        tkmb.showerror("Error", "Invalid Username or Password")

def signup():

    app.withdraw()  # Hide the main window

    signup_window = ctk.CTkToplevel(app)
    signup_window.title("Sign Up")
    signup_window.geometry("1280x720")

    width=700

    signup_frame = ctk.CTkFrame(master=signup_window, width=width)
    signup_frame.pack(pady=20, padx=40, fill='y', expand=False)

    signup_label = ctk.CTkLabel(master=signup_frame, text='Create an Account')
    signup_label.pack(pady=20, padx=18)

    new_username_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Username")
    new_username_entry.pack(pady=20, padx=18)

    new_password_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Password", show="*")
    new_password_entry.pack(pady=20, padx=18)

    confirm_password_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Confirm Password", show="*")
    confirm_password_entry.pack(pady=20, padx=18)

    age_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Age")
    age_entry.pack(pady=20, padx=18)

    designation_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Designation")
    designation_entry.pack(pady=20, padx=18)

    def register_user():
        new_username = new_username_entry.get()
        new_password = new_password_entry.get()
        confirm_password = confirm_password_entry.get()
        age = age_entry.get()
        designation = designation_entry.get()

        if not is_valid_username(new_username):
            tkmb.showwarning(title="Invalid Username", message="Alphanumeric, min 3 chars.")
            return

        if not is_valid_password(new_password):
            tkmb.showwarning(title="Invalid Password", message="Min 8 chars, uppercase, lowercase, digit, special.")
            return

        if not age.isdigit() or int(age) <= 0:
            tkmb.showwarning(title="Invalid Age", message="Age must be a positive number.")
            return

        if not designation:
            tkmb.showwarning(title="Invalid Designation", message="Designation cannot be empty.")
            return

        if new_username in user_data:
            tkmb.showerror(title="Signup Failed", message="Username exists.")
            return

        if new_password != confirm_password:
            tkmb.showerror(title="Signup Failed", message="Passwords do not match.")
            return

        # Store user data as a dictionary
        user_data[new_username] = {
            'password': new_password,
            'age': age,
            'designation': designation
        }

        if save_user_data(user_data):
            tkmb.showinfo(title="Signup Successful", message="Account created!")
            signup_window.destroy()
            app.deiconify()  # Show the main window again

        else:
            tkmb.showerror("Error", "Signup failed.")

    signup_button = ctk.CTkButton(master=signup_frame, text='Sign Up', command=register_user)
    signup_button.pack(pady=20, padx=18)

    # Add "If you already have an account, login instead" label/button
    login_instead_label = ctk.CTkLabel(master=signup_frame, text="Already have an account?")
    login_instead_label.pack(pady=20, padx=18)

    login_instead_button = ctk.CTkButton(master=signup_frame, text="Login", command=lambda: [signup_window.destroy(), app.deiconify()])
    login_instead_button.pack(pady=20, padx=18)
    center_window(signup_window, 1280, 720)
    # Make the main window appear when signup window is closed
    signup_window.protocol("WM_DELETE_WINDOW", lambda: (app.deiconify(), signup_window.destroy()))

ctk_logo = ctk.CTkImage(light_image=logo, dark_image=logo, size=(200,200))
label = ctk.CTkLabel(app, image=ctk_logo, text ="")
label.pack(pady=10)

frame_width = 250
frame_height = 400
frame = ctk.CTkFrame(master=app, width=frame_width, height=frame_height)
frame.pack(pady=10, padx=0, anchor='center')
frame.pack_propagate(False)

label = ctk.CTkLabel(master=frame, text='LOGIN', font=('Arial', 24))
label.pack(pady=12, padx=10)

user_entry = ctk.CTkEntry(master=frame, placeholder_text="Username")
user_entry.pack(pady=20, padx=18)

user_pass = ctk.CTkEntry(master=frame, placeholder_text="Password", show="*")
user_pass.pack(pady=20, padx=18)

# Initialize with remembered username, if any
user_entry.insert(0, remembered_username)

button = ctk.CTkButton(master=frame, text='Login', command=login)
button.pack(pady=20, padx=18)

remember_me_checkbox = ctk.CTkCheckBox(master=frame, text='Remember Me')
remember_me_checkbox.pack(pady=20, padx=18)

# Set initial state of checkbox based on loaded data
remember_me_checkbox.select() if remember_me else remember_me_checkbox.deselect()

signup_button = ctk.CTkButton(master=frame, text='Sign Up', command=signup)
signup_button.pack(pady=20, padx=18)

center_window(app, 1280,720)
app.mainloop()
