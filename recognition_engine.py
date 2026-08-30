"""
recognition_engine.py - ArcFace engine with auto-install, auto-download,
and direct ONNX fallbacks.

On first run:
  1. Checks if buffalo_l model is downloaded.
     If not → downloads it automatically with progress output.
  2. Tries the insightface FaceAnalysis backend.
  3. Falls back to direct onnxruntime embeddings if insightface cannot import.
  4. Falls back to OpenCV DNN if the onnxruntime Windows DLL cannot load.

On Windows the direct ONNX Runtime backend is preferred.  It avoids a costly
InsightFace detector startup and is substantially faster for the app's
single-face webcam flow.
"""

import os, sys, threading, subprocess, zipfile, shutil, importlib
import numpy as np
import cv2

_BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.expanduser("~/.insightface")
MODEL_DIR  = os.path.join(MODEL_ROOT, "models", "buffalo_l")
ZIP_URL    = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ZIP_PATH   = os.path.join(MODEL_ROOT, "models", "buffalo_l.zip")

# InsightFace receives aligned faces and can use a stricter score.  The direct
# ONNX/OpenCV fallback receives a consistently cropped Haar face, so it needs a
# slightly lower threshold to avoid rejecting the same person unnecessarily.
SIMILARITY_THRESHOLD           = 0.45  # kept for compatibility with callers
DIRECT_RECOGNITION_THRESHOLD   = 0.55
INSIGHTFACE_RECOGNITION_THRESHOLD = 0.45
DIRECT_DUPLICATE_THRESHOLD     = 0.60
INSIGHTFACE_DUPLICATE_THRESHOLD = 0.55

_lock        = threading.Lock()
_app         = None
_ort_session = None
_ort_input   = ""
_cv_net      = None
_cv_lock     = threading.Lock()
_backend     = "unavailable"
_initialized = False
_error_msg   = ""
_status      = "not_initialized"   # 'ready' | 'no_model' | 'error'


# ── Runtime package helpers ───────────────────────────────────────────────────

def _format_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _looks_like_onnxruntime_native_error(message: str) -> bool:
    low = message.lower()
    return any(
        part in low
        for part in (
            "unable to import dependency onnxruntime",
            "onnxruntime_pybind11_state",
            "dll load failed",
            "dynamic link library",
        )
    )


def _run_pip(packages: list[str]) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         *packages, "--break-system-packages", "-q"],
        capture_output=False,
    )
    return result.returncode == 0


def _ensure_insightface():
    """Return FaceAnalysis when insightface can be imported, otherwise None."""
    try:
        if os.name == "nt":
            try:
                import msvc_runtime  # noqa: F401
            except Exception:
                pass
        from insightface.app import FaceAnalysis
        return FaceAnalysis, ""
    except Exception as exc:
        first_error = _format_exc(exc)

    print(f"[ArcFace] insightface import failed: {first_error}")
    if _looks_like_onnxruntime_native_error(first_error):
        return None, first_error

    print("[ArcFace] Trying to repair insightface install...")
    repair_packages = ["insightface", "onnxruntime", "setuptools"]
    if os.name == "nt":
        repair_packages.insert(0, "msvc-runtime")
    if not _run_pip(repair_packages):
        return None, first_error

    try:
        importlib.invalidate_caches()
        for name in list(sys.modules):
            if name == "insightface" or name.startswith("insightface."):
                del sys.modules[name]
        if os.name == "nt":
            try:
                import msvc_runtime  # noqa: F401
            except Exception:
                pass
        from insightface.app import FaceAnalysis
        print("[ArcFace] insightface repaired successfully")
        return FaceAnalysis, ""
    except Exception as exc:
        return None, f"{first_error}\nAfter reinstall: {_format_exc(exc)}"


def _ensure_onnxruntime():
    """Return onnxruntime when it can be imported, otherwise None."""
    try:
        if os.name == "nt":
            try:
                import msvc_runtime  # noqa: F401
            except Exception:
                pass
        import onnxruntime as ort
        return ort, ""
    except Exception as exc:
        first_error = _format_exc(exc)

    print(f"[ArcFace] onnxruntime import failed: {first_error}")
    if _looks_like_onnxruntime_native_error(first_error) and os.name == "nt":
        print("[ArcFace] Trying to add Windows runtime DLL support...")
        if _run_pip(["msvc-runtime"]):
            try:
                importlib.invalidate_caches()
                if "onnxruntime" in sys.modules:
                    del sys.modules["onnxruntime"]
                import msvc_runtime  # noqa: F401
                import onnxruntime as ort
                print("[ArcFace] onnxruntime repaired successfully")
                return ort, ""
            except Exception as exc:
                return None, f"{first_error}\nAfter Windows runtime repair: {_format_exc(exc)}"
        return None, first_error

    print("[ArcFace] Trying to repair onnxruntime install...")
    repair_packages = ["onnxruntime"]
    if os.name == "nt":
        repair_packages.insert(0, "msvc-runtime")
    if not _run_pip(repair_packages):
        return None, first_error

    try:
        importlib.invalidate_caches()
        if "onnxruntime" in sys.modules:
            del sys.modules["onnxruntime"]
        if os.name == "nt":
            try:
                import msvc_runtime  # noqa: F401
            except Exception:
                pass
        import onnxruntime as ort
        print("[ArcFace] onnxruntime repaired successfully")
        return ort, ""
    except Exception as exc:
        return None, f"{first_error}\nAfter reinstall: {_format_exc(exc)}"


# ── Step 2: ensure model is downloaded ───────────────────────────────────────

def _model_ready() -> bool:
    required = ["det_10g.onnx", "w600k_r50.onnx"]
    if not os.path.isdir(MODEL_DIR):
        return False
    existing = set(os.listdir(MODEL_DIR))
    return all(f in existing for f in required)


def _download_model() -> bool:
    """Download buffalo_l.zip and extract it. Returns True on success."""
    import requests

    os.makedirs(os.path.join(MODEL_ROOT, "models"), exist_ok=True)

    # Clean up any previous partial download
    if os.path.isdir(MODEL_DIR):
        bad = set(os.listdir(MODEL_DIR))
        if not all(f in bad for f in ["det_10g.onnx", "w600k_r50.onnx"]):
            shutil.rmtree(MODEL_DIR, ignore_errors=True)

    print(f"[ArcFace] Downloading buffalo_l model (~350 MB) …")
    print(f"[ArcFace] Source: {ZIP_URL}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/octet-stream,*/*",
    }

    try:
        r = requests.get(ZIP_URL, stream=True, headers=headers,
                         allow_redirects=True, timeout=120)
        if r.status_code != 200:
            print(f"[ArcFace] Download failed: HTTP {r.status_code}")
            print(f"[ArcFace] Try manually: python download_models.py")
            return False

        total    = int(r.headers.get("content-length", 0))
        received = 0
        with open(ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    received += len(chunk)
                    if total:
                        pct = int(50 * received / total)
                        bar = "█" * pct + "░" * (50 - pct)
                        mb  = received / 1048576
                        tot = total    / 1048576
                        print(f"\r[ArcFace] [{bar}] {mb:.0f}/{tot:.0f} MB",
                              end="", flush=True)
        print()

        # Validate size (must be > 100 MB to be real)
        if os.path.getsize(ZIP_PATH) < 100_000_000:
            print("[ArcFace] Downloaded file too small — likely an error page.")
            os.remove(ZIP_PATH)
            return False

    except Exception as e:
        print(f"[ArcFace] Download error: {e}")
        print(f"[ArcFace] Try: python download_models.py")
        return False

    # Extract
    print("[ArcFace] Extracting …", end=" ", flush=True)
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(MODEL_DIR)
        os.remove(ZIP_PATH)
        print("done ✓")
        return _model_ready()
    except Exception as e:
        print(f"failed: {e}")
        return False


# ── Step 3: load FaceAnalysis ─────────────────────────────────────────────────

def _load_with_insightface(FaceAnalysis) -> tuple[bool, str]:
    global _app, _backend
    try:
        print("[ArcFace] Loading buffalo_l with insightface ...", end=" ", flush=True)
        app = FaceAnalysis(
            name="buffalo_l",
            root=MODEL_ROOT,
            providers=["CPUExecutionProvider"],
        )
        # A 320 detector is much faster on a CPU webcam stream and remains
        # accurate for faces at the dashboard's 640x480 display size.
        app.prepare(ctx_id=-1, det_size=(320, 320))
        _app     = app
        _backend = "insightface"
        print("ready")
        return True, ""
    except Exception as exc:
        return False, _format_exc(exc)


def _load_with_onnxruntime() -> tuple[bool, str]:
    global _ort_session, _ort_input, _backend
    model_path = os.path.join(MODEL_DIR, "w600k_r50.onnx")
    if not os.path.isfile(model_path):
        return False, f"Missing ArcFace embedding model: {model_path}"

    ort, err = _ensure_onnxruntime()
    if ort is None:
        return False, err

    try:
        print("[ArcFace] Loading w600k_r50.onnx directly ...", end=" ", flush=True)
        _ort_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        _ort_input = _ort_session.get_inputs()[0].name
        _ort_session.run(None, {_ort_input: np.zeros((1, 3, 112, 112), dtype=np.float32)})
        _backend = "onnxruntime-direct"
        print("ready")
        return True, ""
    except Exception as exc:
        return False, _format_exc(exc)


def _load_with_opencv_dnn() -> tuple[bool, str]:
    global _cv_net, _backend
    model_path = os.path.join(MODEL_DIR, "w600k_r50.onnx")
    if not os.path.isfile(model_path):
        return False, f"Missing ArcFace embedding model: {model_path}"

    try:
        print("[ArcFace] Loading w600k_r50.onnx with OpenCV DNN ...", end=" ", flush=True)
        net = cv2.dnn.readNetFromONNX(model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        net.setInput(np.zeros((1, 3, 112, 112), dtype=np.float32))
        net.forward()
        _cv_net = net
        _backend = "opencv-dnn-direct"
        print("ready")
        return True, ""
    except Exception as exc:
        return False, _format_exc(exc)


def _load():
    global _error_msg, _status

    # Step 1: model files
    if not _model_ready():
        ok = _download_model()
        if not ok:
            _error_msg = (
                "buffalo_l model not downloaded.\n"
                "Fix: python download_models.py"
            )
            _status = "no_model"
            print(f"[ArcFace] ✗ {_error_msg.splitlines()[0]}")
            return

    # We now default to direct ONNX as the primary backend because the full
    # InsightFace (buffalo_l) RetinaFace detector is too slow (~4 seconds)
    # on many Windows CPUs. Haar Cascade + direct ONNX is much faster.
    
    print("[ArcFace] Attempting direct ONNX ArcFace backend (Fastest) ...")
    ok, onnx_error = _load_with_onnxruntime()
    if ok:
        _error_msg = ""
        _status    = "ready"
        return
    print(f"[ArcFace] ONNX direct failed: {onnx_error}")

    # Fallback 1: OpenCV DNN
    print("[ArcFace] Falling back to OpenCV DNN ArcFace backend.")
    ok, opencv_error = _load_with_opencv_dnn()
    if ok:
        _error_msg = ""
        _status    = "ready"
        return
    print(f"[ArcFace] OpenCV DNN failed: {opencv_error}")

    # Fallback 2: Full InsightFace (slow on CPU)
    insightface_error = ""
    FaceAnalysis, insightface_error = _ensure_insightface()
    if FaceAnalysis is not None:
        ok, err = _load_with_insightface(FaceAnalysis)
        if ok:
            _error_msg = ""
            _status    = "ready"
            return
        insightface_error = f"{insightface_error}\nFaceAnalysis load: {err}".strip()
        print(f"[ArcFace] insightface load failed: {err}")

    _error_msg = (
        "ArcFace could not load with onnxruntime, OpenCV DNN, or insightface.\n"
        f"onnxruntime: {onnx_error or 'not available'}\n"
        f"opencv_dnn: {opencv_error or 'not available'}\n"
        f"insightface: {insightface_error or 'not available'}\n"
        "Fix: run python install.py again, then start python app.py."
    )
    _status = "error"
    print(f"[ArcFace] ✗ {_error_msg.splitlines()[0]}")


def initialize():
    global _initialized
    with _lock:
        if _initialized:
            return
        _load()
        _initialized = True


def is_ready() -> bool:
    return _app is not None or _ort_session is not None or _cv_net is not None


def model_status() -> dict:
    return {
        "ready":       _app is not None or _ort_session is not None or _cv_net is not None,
        "status":      _status,
        "mode":        _backend,
        "error":       _error_msg,
        "model_dir":   MODEL_DIR,
        "files_exist": _model_ready(),
    }


# ── Face embedding ─────────────────────────────────────────────────────────────

def has_insightface() -> bool:
    """Whether the full InsightFace detector/alignment backend is active."""
    initialize()
    return _app is not None


def recognition_threshold() -> float:
    """Return the conservative matching threshold for the active backend."""
    return (INSIGHTFACE_RECOGNITION_THRESHOLD if has_insightface()
            else DIRECT_RECOGNITION_THRESHOLD)


def duplicate_threshold() -> float:
    """Return a stricter threshold for rejecting a duplicate registration."""
    return (INSIGHTFACE_DUPLICATE_THRESHOLD if has_insightface()
            else DIRECT_DUPLICATE_THRESHOLD)


def crop_face(bgr: np.ndarray, box: tuple[int, int, int, int],
              padding: float = 0.20) -> "np.ndarray | None":
    """Create a stable square face crop aligned for ArcFace ONNX model.

    ArcFace expects eyes at ~46% from the top. Haar cascade boxes place
    eyes at ~30% from the top. We expand the box vertically to match the 
    ArcFace template better.
    """
    if bgr is None or bgr.size == 0:
        return None
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
        
    # Scale up the box to provide head context.
    # A multiplier of 1.5 closely maps the Haar box to the 112x112 ArcFace template.
    side = max(w, h) * (1.5 + padding)
    
    # Shift center down so the top edge moves up, pushing the eyes down relative to the crop
    cx, cy = x + w / 2.0, y + h / 2.0 + (side * 0.12)
    
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))
    
    # Safe crop with padding if box exceeds image boundaries
    h_img, w_img = bgr.shape[:2]
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h_img)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w_img)
    
    bgr_pad = cv2.copyMakeBorder(bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(127,127,127))
    
    ny1, ny2 = y1 + pad_top, y2 + pad_top
    nx1, nx2 = x1 + pad_left, x2 + pad_left
    crop = bgr_pad[ny1:ny2, nx1:nx2]
    return crop.copy() if crop.size else None


def get_faces_with_embeddings(bgr: np.ndarray) -> list[dict]:
    """Return normalized InsightFace embeddings and display-space boxes.

    This is deliberately a small public adapter around InsightFace.  The live
    worker uses it so backend-specific Face objects never leak into attendance.
    """
    results: list[dict] = []
    for face in get_faces(bgr):
        try:
            x1, y1, x2, y2 = [int(round(v)) for v in face.bbox[:4]]
            emb = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(emb)
            if norm <= 1e-8:
                continue
            results.append({
                "box": (x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                "embedding": (emb / norm).astype(np.float32),
                "det_score": float(getattr(face, "det_score", 0.0)),
            })
        except Exception as exc:
            print(f"[ArcFace] face result conversion: {exc}")
    return results

def get_faces(bgr: np.ndarray) -> list:
    initialize()
    if _app is None or bgr is None or bgr.size == 0:
        return []
    try:
        return _app.get(bgr) or []
    except Exception as e:
        print(f"[ArcFace] get_faces: {e}")
        return []


def _square_face(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

    h, w = bgr.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(
        bgr, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(127, 127, 127),
    )


def _get_embedding_onnx(roi_bgr: np.ndarray) -> "np.ndarray | None":
    if _ort_session is None or roi_bgr is None or roi_bgr.size == 0:
        return None
    try:
        img = _square_face(roi_bgr)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = (img.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]
        out = _ort_session.run(None, {_ort_input: blob})[0]
        emb = np.asarray(out[0], dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        return (emb / norm).astype(np.float32) if norm > 1e-8 else emb
    except Exception as exc:
        print(f"[ArcFace] onnx embedding: {exc}")
        return None


def _get_embedding_opencv(roi_bgr: np.ndarray) -> "np.ndarray | None":
    if _cv_net is None or roi_bgr is None or roi_bgr.size == 0:
        return None
    try:
        img = _square_face(roi_bgr)
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        with _cv_lock:
            _cv_net.setInput(blob)
            out = _cv_net.forward()
        emb = np.asarray(out[0], dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        return (emb / norm).astype(np.float32) if norm > 1e-8 else emb
    except Exception as exc:
        print(f"[ArcFace] opencv embedding: {exc}")
        return None


def get_embedding(roi_bgr: np.ndarray) -> "np.ndarray | None":
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    initialize()
    if _ort_session is not None and _app is None:
        return _get_embedding_onnx(roi_bgr)
    if _cv_net is not None and _app is None:
        return _get_embedding_opencv(roi_bgr)

    h, w = roi_bgr.shape[:2]
    if h < 48 or w < 48:
        scale   = max(112 / max(h, 1), 112 / max(w, 1))
        roi_bgr = cv2.resize(roi_bgr, (int(w * scale), int(h * scale)))

    faces = get_faces(roi_bgr)
    img_for_center = roi_bgr
    if not faces:
        pad    = max(roi_bgr.shape[:2]) // 3
        padded = cv2.copyMakeBorder(roi_bgr, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=(127, 127, 127))
        faces  = get_faces(padded)
        img_for_center = padded
    if not faces:
        return None

    # Pick the face closest to the center of the crop.
    # This prevents picking a different person who happens to have a 
    # higher detection score when multiple faces appear in the same crop.
    cx, cy = img_for_center.shape[1] / 2.0, img_for_center.shape[0] / 2.0
    best = min(faces, key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2.0 - cx)**2 + ((f.bbox[1] + f.bbox[3]) / 2.0 - cy)**2)
    emb  = best.embedding
    if emb is None:
        return None
    norm = np.linalg.norm(emb)
    return (emb / norm).astype(np.float32) if norm > 1e-8 else emb.astype(np.float32)


# ── Similarity + serialisation ────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)))


def embedding_to_bytes(emb: np.ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()


def bytes_to_embedding(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32).copy()


def find_best_match(query_emb: np.ndarray, index: list,
                    threshold: float | None = None) -> tuple:
    best_entry = None
    best_score = 0.0
    second_entry = None
    second_score = 0.0
    for entry in index:
        score = cosine_sim(query_emb, entry["embedding"])
        if score > best_score:
            second_score = best_score
            second_entry = best_entry
            best_score = score
            best_entry = entry
        elif score > second_score:
            second_score = score
            second_entry = entry
    if threshold is None:
        threshold = recognition_threshold()
    if best_score >= threshold:
        return best_entry, best_score, second_entry, second_score
    return None, best_score, second_entry, second_score


# Load in background thread to prevent startup lag
threading.Thread(target=initialize, daemon=True).start()
