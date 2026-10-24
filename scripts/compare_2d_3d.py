#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path and delegate to root compare_2d_3d
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from compare_2d_3d import main

if __name__ == "__main__":
    main()
