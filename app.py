import os
import sys
from pathlib import Path

# Add src and root to Python path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import uvicorn
from demo.backend.api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting PRISM Server on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
