#!/usr/bin/env python3
"""
cx_Freeze setup script for BloomSentryMain application
Run with: python setup.py build
"""

import sys
import os
from cx_Freeze import setup, Executable

# Determine base for executable (GUI vs Console)
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Use "Console" for debugging

# Define packages to include
packages = [
    "customtkinter",
    "tkinter",
    "PIL",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "scipy",
    "geopandas",
    "folium",
    "cryptography",
    "dateutil",
    # Removed built-in modules that don't need to be explicitly included
    "openpyxl",  # Often needed for Excel file handling with pandas
    "xlrd",      # For reading older Excel files
]

# Define modules to exclude (optional - to reduce size)
excludes = [
    "test",
    "unittest",
    "email",
    "http",
    "urllib",
    "xml",
    "pydoc_data",
    "doctest",
    "distutils",
    "setuptools"
]

# Include additional files and directories
include_files = []

# Check and add files/directories if they exist
file_mappings = [
    ("Icons/", "Icons/"),
    ("train/merged_stations.xlsx", "train/merged_stations.xlsx"),
    ("user_data_directory/", "user_data_directory/"),
    ("DevPics/", "DevPics/"),
    ("heatmapper/", "heatmapper/"),
]

for src, dst in file_mappings:
    if os.path.exists(src):
        include_files.append((src, dst))
    else:
        print(f"Warning: {src} not found, skipping...")

# Check if icon file exists
icon_path = "Icons/AppIcon.ico"  # Use .ico for Windows
if not os.path.exists(icon_path):
    icon_path = "Icons/AppIcon.png"  # Fallback to PNG
    if not os.path.exists(icon_path):
        icon_path = None
        print("Warning: No icon file found")

# Build options
build_exe_options = {
    "packages": packages,
    "excludes": excludes,
    "include_files": include_files,
    "optimize": 2,  # Optimize bytecode
    "build_exe": "build/BloomSentryMain",  # Output directory
    "zip_include_packages": "*",  # Compress packages to reduce size
    "zip_exclude_packages": [],
}

# Create executable configuration
executables = [
    Executable(
        "login.py",  # Your main Python file
        base=base,
        target_name="BloomSentryMain.exe" if sys.platform == "win32" else "BloomSentryMain",
        icon=icon_path,
        shortcut_name="BloomSentry",
        shortcut_dir="DesktopFolder" if sys.platform == "win32" else None,
    )
]

# Setup configuration
setup(
    name="BloomSentryMain",
    version="1.0.0",
    description="Water Quality Monitoring and Prediction System",
    author="Your Name",
    options={"build_exe": build_exe_options},
    executables=executables
)