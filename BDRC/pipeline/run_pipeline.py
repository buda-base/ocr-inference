#!/usr/bin/env python3
"""
Wrapper script to run the pipeline as a module.
This allows running from within the pipeline directory.
"""
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from BDRC.pipeline.main import main
    main()


