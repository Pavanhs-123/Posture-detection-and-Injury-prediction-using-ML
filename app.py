"""
Root entry point for Streamlit Badminton AI Dashboard.
Delegates to app/streamlit_app.py.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Execute main Streamlit app module
streamlit_app_path = root_dir / "app" / "streamlit_app.py"
with open(streamlit_app_path, "r", encoding="utf-8") as f:
    code = f.read()
exec(code, globals())
