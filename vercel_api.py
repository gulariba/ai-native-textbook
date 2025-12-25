# vercel_api.py
# Entry point for Vercel deployment
import sys
from os.path import dirname, abspath

# Add the current directory to the path
sys.path.append(abspath(dirname(__file__)))

# Import the main app instance
from main import app_instance

# Vercel expects the application object to be named 'app'
app = app_instance