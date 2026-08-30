"""
face_register.py v8 — ArcFace-only registration.
Camera opens immediately so feed is always live.
Gives clear error if model not downloaded.
"""

import cv2, os, shutil, threading, time
from typing import Iterator
import numpy as np

import database as db
import recognition_engine as eng

_BASE        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(_BASE, "dataset")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

SAMPLE_COUNT      = 20          # 20 diverse samples required
CAPTURE_SIMILARITY_THRESHOLD = 0.97  # must move head slightly
FACE_SIZE        = (200, 200)
CAPTURE_EVERY     = 3           
DETECT_EVERY     = 2
DETECT_SCALE     = 0.5
JPEG_QUALITY     = 70

# ── Shared state ───────────────────────────────────────────────────────────────
_reg_lock     = threading.Lock()
_reg_frame:   bytes | None = None
_reg_progress: int         = 0
_reg_total:   int          = SAMPLE_COUNT
_reg_status:  str          = "idle"
_reg_dup_name: str         = ""
_reg_message: str          = ""


def _set_frame(buf: bytes | None):
    global _reg_frame
    with _reg_lock:
        _reg_frame = buf


def get_reg_status() -> dict:
    return {
        "status":   _reg_status,
        "captured": _reg_progress,
        "total":    _reg_total,
        "pct":      int(100 * _reg_progress / _reg_total) if _reg_total else 0,
        "dup_name": _reg_dup_name,
        "message":  _reg_message,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _open_camera() -> cv2.VideoCapture | None:
    """Open a camera that can deliver frames, with Windows backend fallback."""
    backends = [cv2.CAP_ANY]
    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]

    for backend in backends:
        for idx in (0, 1):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            ready = False
            for _ in range(4):
                ret, _ = cap.read()
                if ret:
                    ready = True
                    break
            if ready:
                print(f"[REG] Camera {idx} opened.")
                return cap
            cap.release()
    return None


def _enc(frame: np.ndarray) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _overlay(frame, line1, line2="", c1=(0,255,100), c2=(180,180,180)):
    cv2.putText(frame, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c1, 2)
    if line2:
        cv2.putText(frame, line2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, c2, 1)
    w = frame.shape[1]
    cv2.putText(frame, "ArcFace", (w-82, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80,255,160), 1)


def _error_frame(msg: str) -> bytes:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (20, 22, 30)
    words, lines, line = msg.split(), [], ""
    for w in words:
        if len(line)+len(w)+1 > 62:
            lines.append(line); line = w
        else:
            line = (line+" "+w).strip()
    if line: lines.append(line)
    y = 210
    for l in lines:
        cv2.putText(img, l, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80,160,255), 1)
        y += 30
    return _enc(img) or b""


def _usable_face(roi: np.ndarray | None) -> bool:
    """Keep tiny, black, and extremely blurry captures out of a profile."""
    if roi is None or roi.size == 0 or min(roi.shape[:2]) < 72:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return 25.0 <= brightness <= 235.0 and sharpness >= 8.0


# ── Registration ───────────────────────────────────────────────────────────────

def register_user_web(name: str, stop_event: threading.Event) -> int:
    global _reg_progress, _reg_status, _reg_total, _reg_dup_name, _reg_message

    _reg_progress = 0
    _reg_total    = SAMPLE_COUNT
    _reg_dup_name = ""
    _reg_message  = ""
    _reg_status   = "starting"

    # ── Open camera FIRST — feed must be live before any checks ──────────────
    cap = _open_camera()
    if cap is None:
        _reg_message = "Camera not found. Check webcam is connected."
        _reg_status  = "error"
        _set_frame(_error_frame("Camera not found. Check webcam is connected."))
        time.sleep(3); _set_frame(None)
        return -1

    existing_user = db.get_user_by_name(name)
    if existing_user:
        label = existing_user["label"]
        print(f"[REG] Found existing user '{name}', updating ID {existing_user['id']}")
    else:
        label = db.next_label()
    save_dir = os.path.join(DATASET_DIR, str(label))
    os.makedirs(save_dir, exist_ok=True)

    def cleanup(success=False):
        try: cap.release()
        except Exception: pass
        _set_frame(None)
        if not success:
            shutil.rmtree(save_dir, ignore_errors=True)

    # ── Check ArcFace model ───────────────────────────────────────────────────
    if not eng.is_ready():
        st = eng.model_status()
        err = st["error"]
        _reg_message = err
        _reg_status  = "model_error"
        print(f"[REG] ArcFace not ready: {err}")

        # Show error on live feed for 5s with helpful instructions
        lines = [
            "ArcFace model not ready.",
            st["error"].split("\n")[0][:52],
            "Fix: python download_models.py",
        ] if not st["files_exist"] else [
            "ArcFace model loading failed.",
            st["error"].split("\n")[0][:52],
            "Fix: python install.py",
        ]
        deadline = time.time() + 5
        while time.time() < deadline and not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                for i, l in enumerate(lines):
                    cv2.putText(frame, l, (10, 50+i*26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,100,240), 2)
                _set_frame(_enc(frame))
            time.sleep(0.05)
        cleanup(success=False)
        return -1

    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # Load existing embeddings for duplicate check
    rows     = db.get_all_embeddings()
    existing = [
        {"name": r["name"], "emb": eng.bytes_to_embedding(r["embedding"])}
        for r in rows if r.get("embedding")
    ]

    if stop_event.is_set():
        _reg_status = "error"; _reg_message = "Cancelled."
        cleanup(); return -1

    # ── Phase 1: Capture face samples ─────────────────────────────────────────
    _reg_status = "running"
    count = 0; frame_idx = 0; faces = ()
    embed_samples: list[np.ndarray] = []
    rejected_count = 0
    instruction = "Move your head slightly left/right and change your angle."
    print(f"[REG] Capturing '{name}' label={label}")

    while count < SAMPLE_COUNT and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: time.sleep(0.03); continue
        frame_idx += 1

        if frame_idx % DETECT_EVERY == 0 and not cascade.empty():
            small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            raw   = cascade.detectMultiScale(gray,1.1,5,minSize=(40,40))
            inv   = 1.0/DETECT_SCALE
            faces = (tuple((int(x*inv),int(y*inv),int(w*inv),int(h*inv))
                           for x,y,w,h in raw) if len(raw) else ())

        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,100),2)
            if frame_idx % CAPTURE_EVERY == 0 and count < SAMPLE_COUNT:
                roi = eng.crop_face(frame, (x, y, w, h))
                if _usable_face(roi):
                    # 1. Extract embedding in real-time
                    emb = eng.get_embedding(roi)
                    if emb is not None:
                        # 2. Check similarity against previously accepted embeddings in this session
                        max_sim = max([eng.cosine_sim(emb, e) for e in embed_samples], default=0.0)
                        if max_sim > CAPTURE_SIMILARITY_THRESHOLD:
                            rejected_count += 1
                            instruction = "Face too similar — move your head slightly."
                        else:
                            # 3. Accept sample
                            instruction = "Good! Keep changing your angle."
                            embed_samples.append(emb)
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            cv2.imwrite(os.path.join(save_dir,f"{count:03d}.jpg"), cv2.resize(gray_roi, FACE_SIZE))
                            count += 1; _reg_progress = count
                    else:
                        instruction = "Embedding failed — ensure good lighting."

        pct   = int(100*count/SAMPLE_COUNT)
        bar_w = int(frame.shape[1]*pct/100)
        cv2.rectangle(frame,(0,frame.shape[0]-6),(bar_w,frame.shape[0]),(0,200,80),cv2.FILLED)
        if len(faces) > 1:
            display_instr = "Only one face should be in the frame"
        elif not faces:
            display_instr = "Face not detected clearly"
        else:
            display_instr = instruction
            
        _overlay(frame, f"Capturing face samples: {count} / {SAMPLE_COUNT}", display_instr)
        _set_frame(_enc(frame))

    if stop_event.is_set() or count < SAMPLE_COUNT:
        cleanup(success=False)
        _reg_status = "error"; _reg_message = f"Cancelled ({count}/{SAMPLE_COUNT})."
        print(f"[REG] Cancelled. Valid samples: {count}, Rejected duplicates: {rejected_count}")
        return -1
    
    print(f"[REG] Captured {count} valid samples. Rejected duplicates: {rejected_count}")

    # ── Phase 3: Create Prototype Embedding ───────────────────────────────────
    _reg_status = "embedding"
    _reg_message = "Creating representative prototype embedding..."
    
    emb_list = embed_samples
    if not emb_list:
        cleanup(success=False)
        _reg_status = "error"
        _reg_message = "No valid embeddings were captured."
        return -1

    np.save(os.path.join(save_dir, "raw_embeddings.npy"), np.array(emb_list))
    
    # Calculate spherical centroid (L2-normalized mean)
    mean_embedding = np.mean(emb_list, axis=0)
    norm = np.linalg.norm(mean_embedding)
    prototype = (mean_embedding / norm).astype(np.float32) if norm > 1e-8 else mean_embedding.astype(np.float32)
    
    # Evaluate prototype accuracy against its own samples
    sims = [eng.cosine_sim(prototype, e) for e in emb_list]
    avg_self_sim = sum(sims) / len(sims)
    print(f"[REG] Prototype self-similarity (mean over 20 samples): {avg_self_sim:.4f}")

    if existing:
        dup_name = ""
        dup_score = 0.0
        for e in existing:
            # Skip checking against the user's own old embedding during an update
            if existing_user and e["name"].lower() == name.lower():
                continue
            sim = eng.cosine_sim(prototype, e["emb"])
            if sim > dup_score:
                dup_score = sim
                dup_name = e["name"]
        
        print(f"[REG] Max similarity against other users: {dup_score:.4f} (User: {dup_name or 'None'})")
        if dup_score >= eng.duplicate_threshold():
            cleanup(success=False)
            _reg_dup_name = dup_name
            _reg_message  = f"Already registered as '{dup_name}'"
            _reg_status   = "duplicate"
            print(f"[REG] Duplicate: '{name}' matched '{dup_name}' ({dup_score:.2f}).")
            return -1



    cleanup(success=True)

    # ── Phase 4: Persist ──────────────────────────────────────────────────────
    try:
        if existing_user:
            uid = existing_user["id"]
        else:
            uid = db.add_user(name, label)
        db.save_embedding(uid, eng.embedding_to_bytes(prototype))
    except Exception as exc:
        _reg_status = "error"
        _reg_message = f"Could not save user: {exc}"
        shutil.rmtree(save_dir, ignore_errors=True)
        print(f"[REG] Save failed for '{name}': {exc}")
        return -1

    _reg_status = "done"
    _reg_message = f"'{name}' registered. {len(emb_list)} embeddings averaged."
    print(f"[REG] Done: '{name}' label={label}  {len(emb_list)} embeddings.")
    return label


# ── MJPEG stream ───────────────────────────────────────────────────────────────

def generate_reg_frames(stop_event: threading.Event) -> Iterator[bytes]:
    TERMINAL = {"done","error","duplicate","model_error","idle"}
    deadline  = time.time() + 90

    while time.time() < deadline:
        with _reg_lock:
            frame = _reg_frame
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(1/25)
            continue
        if stop_event.is_set() or _reg_status in TERMINAL:
            return
        time.sleep(0.04)
