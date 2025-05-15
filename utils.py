from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64


def decrypt_data(encrypted_data, encryption_key):
    f = Fernet(encryption_key)
    decrypted_data = f.decrypt(encrypted_data).decode()
    return decrypted_data
    pass


def generate_key(password, salt):
    password_encoded = password.encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password_encoded))
    return key


MASTER_KEY = b"YourMasterEncryptionKey"  # Example key