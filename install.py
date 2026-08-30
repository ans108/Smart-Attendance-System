"""
install.py — One-command setup. Run: python install.py
Uses the SAME python that runs this script for all pip commands.
"""
import subprocess, sys

PY  = sys.executable   # always the correct python
PIP = [PY, "-m", "pip", "install", "--break-system-packages", "-q"]

def run(packages, label):
    print(f"\n  Installing {label}…")
    r = subprocess.run(PIP + packages)
    return r.returncode == 0

def print_error(prefix, text):
    lines = (text or "").strip().splitlines()
    if not lines:
        return
    print(prefix)
    for line in lines[:8]:
        print(f"     {line}")

def verify_engine():
    print("\n  Checking ArcFace backend…")
    code = (
        "import recognition_engine as eng\n"
        "st = eng.model_status()\n"
        "print('__FACEATTEND_READY__', st['ready'], st['mode'])\n"
        "if not st['ready']:\n"
        "    print(st['error'])\n"
        "    raise SystemExit(1)\n"
    )
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    output = (r.stdout or "") + (r.stderr or "")
    ready_line = next((line for line in output.splitlines()
                       if line.startswith("__FACEATTEND_READY__")), "")
    if ready_line:
        parts = ready_line.split(maxsplit=2)
        if len(parts) == 3 and parts[1] == "True":
            print(f"  ✓  ArcFace backend ready: {parts[2]}")
            return True
    print("  ✗  ArcFace backend still not ready.")
    print_error("     Engine output:", output.replace("__FACEATTEND_READY__", ""))
    return False

print("""
╔══════════════════════════════════════════════╗
║       FaceAttend — Setup                     ║
╚══════════════════════════════════════════════╝
""")
print(f"  Python: {PY}")
print(f"  Version: {sys.version.split()[0]}")

# Core
run(["opencv-contrib-python", "numpy", "flask",
     "pillow", "scikit-learn", "requests"], "core packages")

# TTS optional
run(["pyttsx3"], "text-to-speech (optional)")

# ArcFace
print("\n  Installing ArcFace packages…")
ok = run(["setuptools", "msvc-runtime", "insightface", "onnxruntime"], "ArcFace runtime")

# Verify insightface actually imports
if ok:
    r = subprocess.run([PY, "-c", "from insightface.app import FaceAnalysis; import insightface; print('OK', insightface.__version__)"],
                       capture_output=True, text=True)
    if "OK" in r.stdout:
        print(f"  ✓  insightface {r.stdout.strip().split()[-1]} importable")
    else:
        print("  !  insightface installed but not importable.")
        print_error("     Import error:", r.stderr or r.stdout)
        r2 = subprocess.run([PY, "-c", "import msvc_runtime; import onnxruntime as ort; print('OK', ort.__version__)"],
                            capture_output=True, text=True)
        if "OK" in r2.stdout:
            print(f"  ✓  onnxruntime {r2.stdout.strip().split()[-1]} importable")
            print("     The app will use the direct ArcFace ONNX fallback.")
        else:
            print("  ✗  onnxruntime is also not importable.")
            print_error("     Import error:", r2.stderr or r2.stdout)

# Download model
print()
print("  Downloading ArcFace buffalo_l model (~350 MB)…")
r = subprocess.run([PY, "download_models.py"])

verify_engine()

print(f"""
╔══════════════════════════════════════════════╗
║  Done! Start with:  python app.py            ║
║  Dashboard:  http://127.0.0.1:5000           ║
║  Admin:      http://127.0.0.1:5000/admin     ║
║  Login:      admin / admin123                ║
╚══════════════════════════════════════════════╝
""")
