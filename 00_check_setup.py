"""Step 0 - Check that Python, the libraries, and the data files are ready.

How to use in IDLE: File > Open this file, then Run > Run Module (F5).
If a library is missing, set AUTO_INSTALL = True below and run again.
"""
import sys
import importlib
import subprocess
from pathlib import Path

AUTO_INSTALL = False        # change to True to let this script install missing libraries
REQUIRED = {"pandas": "pandas", "numpy": "numpy", "scipy": "scipy",
            "matplotlib": "matplotlib", "openpyxl": "openpyxl"}

BASE = Path(__file__).resolve().parents[1]
print("Python version :", sys.version.split()[0])
print("Python program :", sys.executable)
print("Project folder :", BASE)
print()

missing = []
for module, pip_name in REQUIRED.items():
    try:
        mod = importlib.import_module(module)
        print(f"[OK]      {module} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"[MISSING] {module}")
        missing.append(pip_name)

if missing and AUTO_INSTALL:
    print("\nInstalling:", ", ".join(missing))
    result = subprocess.run([sys.executable, "-m", "pip", "install", *missing],
                            capture_output=True, text=True)
    print(result.stdout[-1500:])
    print(result.stderr[-1500:])
    print("Now close IDLE, reopen it, and run this file again to confirm.")
elif missing:
    print("\nSome libraries are missing. Either set AUTO_INSTALL = True above,")
    print("or open a terminal (Windows: Command Prompt; Mac: Terminal) and run:")
    print(f'  "{sys.executable}" -m pip install {" ".join(missing)}')

print()
raw = BASE / "data" / "raw" / "Golfzon_Raw_data.xlsx"
clean = BASE / "data" / "clean" / "clean.csv"
print("[OK]      clean.csv found" if clean.exists() else "[MISSING] data/clean/clean.csv")
print("[OK]      raw Excel found (you can run step 01)" if raw.exists()
      else "[INFO]    raw Excel not found: skip step 01 and start from step 02 using clean.csv")
