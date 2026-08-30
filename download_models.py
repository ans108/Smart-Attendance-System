"""
download_models.py — Downloads the ArcFace buffalo_l model pack.
Run this ONCE before starting the app:  python download_models.py

Downloads ~350MB from GitHub to:  insightface_models/models/buffalo_l/
"""

import os, sys, zipfile, shutil
import requests
import io

# Force UTF-8 encoding for standard output on Windows to support box-drawing characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL   = "https://github.com/deepinsight/insightface/releases/download/v0.7"
MODEL_NAME = "buffalo_l"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(BASE_DIR, "insightface_models", "models")
MODEL_DIR  = os.path.join(MODEL_ROOT, MODEL_NAME)
ZIP_PATH   = os.path.join(MODEL_ROOT, f"{MODEL_NAME}.zip")


def check_already_downloaded() -> bool:
    """Return True if buffalo_l is already fully downloaded."""
    required = ["det_10g.onnx", "w600k_r50.onnx"]
    if not os.path.isdir(MODEL_DIR):
        return False
    existing = os.listdir(MODEL_DIR)
    return all(f in existing for f in required)


def download_with_progress(url: str, dest: str) -> bool:
    """Download url to dest with a progress bar. Returns True on success."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        "Accept": "application/octet-stream,*/*",
    }
    try:
        r = requests.get(url, stream=True, headers=headers,
                         allow_redirects=True, timeout=60)
        if r.status_code != 200:
            print(f"\n  ✗  HTTP {r.status_code} — download failed.")
            print(f"     Try downloading manually from:\n     {url}")
            print(f"     Save to: {dest}")
            return False

        total     = int(r.headers.get("content-length", 0))
        received  = 0
        bar_width = 40

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    received += len(chunk)
                    if total:
                        pct   = received / total
                        filled = int(bar_width * pct)
                        bar   = "█" * filled + "░" * (bar_width - filled)
                        mb_done  = received / 1024 / 1024
                        mb_total = total    / 1024 / 1024
                        print(f"\r  [{bar}] {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)
        print()
        return True

    except requests.exceptions.ConnectionError:
        print("\n  ✗  Connection failed. Check your internet connection.")
        return False
    except Exception as e:
        print(f"\n  ✗  Error: {e}")
        return False


def extract_zip(zip_path: str, dest_dir: str) -> bool:
    """Extract zip_path into dest_dir. Returns True on success."""
    try:
        print(f"  Extracting…", end=" ", flush=True)
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        print("done.")
        return True
    except zipfile.BadZipFile:
        print("✗  Bad zip file — download may be corrupted.")
        return False
    except Exception as e:
        print(f"✗  {e}")
        return False


def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   FaceAttend — ArcFace Model Download               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    if check_already_downloaded():
        print("  ✓  buffalo_l model already downloaded!")
        print(f"     Location: {MODEL_DIR}")
        print()
        _list_files()
        print()
        print("  You can now run:  python app.py")
        return 0

    print(f"  Model:    InsightFace buffalo_l (ArcFace R50)")
    print(f"  Size:     ~350 MB")
    print(f"  Dest:     {MODEL_DIR}")
    print(f"  Source:   {BASE_URL}/{MODEL_NAME}.zip")
    print()

    # Download
    url = f"{BASE_URL}/{MODEL_NAME}.zip"
    print(f"  Downloading buffalo_l.zip…")
    ok = download_with_progress(url, ZIP_PATH)
    if not ok:
        _manual_instructions()
        return 1

    # Verify zip size (should be ~350MB)
    zip_size = os.path.getsize(ZIP_PATH)
    if zip_size < 10_000_000:
        print(f"  ✗  Downloaded file too small ({zip_size} bytes) — likely an error page.")
        os.remove(ZIP_PATH)
        _manual_instructions()
        return 1

    # Extract
    ok = extract_zip(ZIP_PATH, MODEL_DIR)
    if not ok:
        _manual_instructions()
        return 1

    # Cleanup zip
    try:
        os.remove(ZIP_PATH)
    except Exception:
        pass

    # Verify
    if not check_already_downloaded():
        print("  ✗  Extraction incomplete — expected ONNX files not found.")
        _manual_instructions()
        return 1

    print()
    print("  ✓  buffalo_l model downloaded and extracted!")
    _list_files()
    print()
    print("  Run:  python app.py")
    print()
    return 0


def _list_files():
    if os.path.isdir(MODEL_DIR):
        for f in sorted(os.listdir(MODEL_DIR)):
            size = os.path.getsize(os.path.join(MODEL_DIR, f))
            print(f"     {f}  ({size/1024/1024:.1f} MB)")


def _manual_instructions():
    print()
    print("  ─────────────────────────────────────────────────")
    print("  MANUAL DOWNLOAD INSTRUCTIONS:")
    print()
    print("  1. Open this URL in your browser:")
    print(f"     {BASE_URL}/{MODEL_NAME}.zip")
    print()
    print("  2. Save the file to:")
    print(f"     {MODEL_ROOT}")
    print()
    print("  3. Run this script again to extract it:")
    print("     python download_models.py")
    print("  ─────────────────────────────────────────────────")
    print()


if __name__ == "__main__":
    sys.exit(main())
