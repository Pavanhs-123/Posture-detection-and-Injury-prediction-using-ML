
"""
Root entry point for Streamlit Badminton AI Dashboard.
Delegates to app/streamlit_app.py.
"""

import sys
from pathlib import Path

# The project root is the directory containing this file.
project_root = Path(__file__).resolve().parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Execute the dashboard with its own correct __file__ path.
streamlit_app_path = project_root / "app" / "streamlit_app.py"

with open(streamlit_app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(streamlit_app_path), "exec")

globals()["__file__"] = str(streamlit_app_path)
exec(code, globals())
