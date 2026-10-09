import os
import sys

# Ensure the project root directory is in sys.path so modules and models can be found
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

# Vercel serverless functions look for a WSGI application named `app`
# or a handler function.
if __name__ == "__main__":
    app.run(debug=True)
