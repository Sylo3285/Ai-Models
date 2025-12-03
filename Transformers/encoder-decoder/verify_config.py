import sys
import os

print("Verifying imports...")

try:
    from config import Config
    print(f"Config imported successfully. Device: {Config.DEVICE}")
except ImportError as e:
    print(f"Failed to import Config: {e}")
    sys.exit(1)

try:
    import train_csv
    print("train_csv imported successfully")
except ImportError as e:
    print(f"Failed to import train_csv: {e}")
    sys.exit(1)

try:
    import inference_csv
    print("inference_csv imported successfully")
except ImportError as e:
    print(f"Failed to import inference_csv: {e}")
    sys.exit(1)

try:
    import chat
    print("chat imported successfully")
except ImportError as e:
    print(f"Failed to import chat: {e}")
    sys.exit(1)

try:
    import analyzer
    print("analyzer imported successfully")
except ImportError as e:
    print(f"Failed to import analyzer: {e}")
    sys.exit(1)

try:
    import downloader
    print("downloader imported successfully")
except ImportError as e:
    print(f"Failed to import downloader: {e}")
    sys.exit(1)

print("\nAll imports successful!")
print(f"Data path from config: {Config.CSV_PATH}")
