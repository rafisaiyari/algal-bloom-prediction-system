import os
import csv
import datetime
from pathlib import Path

# Define constants
AUDIT_DIR = "audit_logs"
AUDIT_FILE = "user_activity.csv"

class AuditLogger:
    def __init__(self):
        """Initialize the audit logger and create necessary directories/files."""
        self.audit_dir = Path(AUDIT_DIR)
        self.audit_file = self.audit_dir / AUDIT_FILE
        
        # Create audit directory if it doesn't exist
        if not self.audit_dir.exists():
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the CSV file with headers if it doesn't exist
        if not self.audit_file.exists():
            with open(self.audit_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'username', 'user_type', 'action', 'details'])
    
    def log_activity(self, username, user_type, action, details=""):
        """
        Log user activity to the audit CSV file.
        
        Args:
            username (str): The username of the user
            user_type (str): The type of user (regular, master, etc.)
            action (str): The action being performed (login, logout, etc.)
            details (str, optional): Additional details about the action
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open(self.audit_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, username, user_type, action, details])
            return True
        except Exception as e:
            print(f"Error logging activity: {e}")
            return False
    
    def log_login(self, username, user_type):
        """Log a user login event."""
        return self.log_activity(username, user_type, "login")
    
    def log_logout(self, username, user_type):
        """Log a user logout event."""
        return self.log_activity(username, user_type, "logout")
    
    def log_failed_login(self, username):
        """Log a failed login attempt."""
        return self.log_activity(username, "unknown", "failed_login", "Authentication failed")
    
    def log_system_event(self, username, user_type, event_type, details=""):
        """Log a system event."""
        return self.log_activity(username, user_type, event_type, details)


# Singleton pattern - single instance for the whole application
_audit_logger = None

def get_audit_logger():
    """Get or create the audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger