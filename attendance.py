
"""
attendance.py v8 — ArcFace-only live recognition.

Key fixes over v7:
  - Dedicated camera-reader thread: cap.read() never blocks the main loop.
    Eliminates the 4.4 FPS camera lag seen on Windows/DSHOW cameras.
  - Recognition worker now sends the FULL (half-res) frame to insightface
    instead of a tight cropped ROI.  Insightface's RetinaFace detector
    works correctly on a full scene; it fails on a tight face crop which
    is why recognition was stuck on "Scanning" indefinitely.
  - JPEG stream rate is capped independently of the main processing loop.
  - Haar cascade scaleFactor 1.1 → 1.2 (faster, still reliable).
"""

import cv2, os, queue, threading, time
from collections import deque
from typing import Iterator
import numpy as np

import database as db
import recognition_engine as eng
from time_policy import classify_arrival, load_config as load_time_cfg

CASCADE_PATH           = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MARK_COOLDOWN_SEC      = 10
FRAME_WIDTH            = 640
FRAME_HEIGHT           = 480
STREAM_FPS             = 20          # JPEG encode cap (independent of camera FPS)
JPEG_QUALITY           = 60          # lower → faster encode, still acceptable
MIN_FACE_PX            = 28          # in the reduced detection frame
DETECT_EVERY           = 2           # responsive face box without overloading CPU
DETECT_SCALE           = 0.5
RECOGNIZE_INTERVAL_SEC = 0.32        # never queue more work than the CPU can finish
RESULT_TTL_SEC         = 1.5         # stale matches must disappear quickly
MAX_RESULT_CACHE       = 10
RECOG_FRAME_SCALE      = 0.5         # scale down frame before sending to insightface
CAMERA_STALL_SEC       = 3.0

_frame_lock   = threading.Lock()
_output_frame: bytes | None = None
event_queue   = queue.Queue(maxsize=30)
_banners:     deque         = deque(maxlen=4)
_perf_lock    = threading.Lock()
_perf_state   = {"fps": 0.0, "recognition_ms": 0.0}
_camera_lock  = threading.Lock()
_camera_state = {"state": "idle", "message": "Camera inactive", "last_frame": 0.0}

# ── TTS ───────────────────────────────────────────────────────────────────────
try:
    import pyttsx3 as _px
    _tts = _px.init()
    _tts.setProperty("rate", 150)
    _TTS_OK = True
except Exception:
    _tts = None; _TTS_OK = False

_tts_q = queue.Queue(maxsize=6)

def _tts_worker():
    while True:
        p = _tts_q.get()
        if p is None: break
        if _TTS_OK and _tts:
            try: _tts.say(p); _tts.runAndWait()
            except Exception: pass

threading.Thread(target=_tts_worker, daemon=True).start()

def _speak(t):
    if _TTS_OK:
        try: _tts_q.put_nowait(t)
        except queue.Full: pass

def _push_event(kind, name="", sim=0.0, status="on_time", late=0):
    try:
        event_queue.put_nowait({
            "type": kind, "name": name,
            "time": time.strftime("%H:%M:%S"),
            "confidence": round(sim * 100, 1),
            "status": status, "late_minutes": late,
        })
    except queue.Full: pass

# ── Embedding index ────────────────────────────────────────────────────────────
_embed_idx:  list[dict]     = []
_embed_lock: threading.Lock = threading.Lock()

def reload_embeddings() -> int:
    global _embed_idx
    rows  = db.get_all_embeddings()
    index = [
        {"user_id": r["user_id"], "name": r["name"],
         "label": r["label"], "embedding": eng.bytes_to_embedding(r["embedding"])}
        for r in rows if r.get("embedding")
    ]
    with _embed_lock:
        _embed_idx = index
    print(f"[ATT] Embeddings loaded: {len(index)}")
    return len(index)

# ── Drawing ────────────────────────────────────────────────────────────────────

def _brackets(f, x, y, w, h, col, thick=2):
    bl = max(16, w // 5)
    for (x1, y1, x2, y2) in [
        (x, y, x+bl, y), (x, y, x, y+bl),
        (x+w, y, x+w-bl, y), (x+w, y, x+w, y+bl),
        (x, y+h, x+bl, y+h), (x, y+h, x, y+h-bl),
        (x+w, y+h, x+w-bl, y+h), (x+w, y+h, x+w, y+h-bl),
    ]:
        cv2.line(f, (x1, y1), (x2, y2), col, thick)

def _badge(f, text, x, y, col):
    fn, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, _h), _ = cv2.getTextSize(text, fn, sc, th)
    pad = 5; top = max(0, y - _h - pad * 2 - 2)
    cv2.rectangle(f, (x, top), (x + tw + pad * 2, y), col, cv2.FILLED)
    cv2.putText(f, text, (x + pad, y - pad), fn, sc, (10, 10, 10), th)

def _draw_banners(frame):
    now = time.time(); fh = frame.shape[0]
    for i, (msg, exp, col) in enumerate(
            reversed([(m, e, c) for m, e, c in _banners if e > now])):
        y = fh - 14 - i * 36
        fn, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        (tw, _h), _ = cv2.getTextSize(msg, fn, sc, th)
        cv2.rectangle(frame, (6, y - _h - 8), (6 + tw + 14, y + 6), col, cv2.FILLED)
        cv2.putText(frame, msg, (13, y), fn, sc, (10, 10, 10), th)


def runtime_status() -> dict:
    with _perf_lock:
        status = dict(_perf_state)
    with _camera_lock:
        status.update(_camera_state)
    return status


def _set_perf(**updates):
    with _perf_lock:
        _perf_state.update(updates)


def _set_camera_state(state: str, message: str, last_frame: float | None = None) -> None:
    with _camera_lock:
        _camera_state["state"] = state
        _camera_state["message"] = message
        if last_frame is not None:
            _camera_state["last_frame"] = last_frame


# ── Camera reader thread ───────────────────────────────────────────────────────

def _camera_reader(cap, buf: list, buf_lock: threading.Lock,
                   stop_event: threading.Event) -> None:
    """Dedicated thread: reads frames as fast as the camera delivers them.

    Stores only the *latest* frame so the main processing loop always gets
    a fresh image without ever blocking on cap.read().
    """
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(
                    frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA
                )
            with buf_lock:
                buf[0] = frame
            _set_camera_state("running", "Camera is live", time.time())
        else:
            time.sleep(0.005)


# ── Recognition worker ─────────────────────────────────────────────────────────

def _box_iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter    = iw * ih
    area     = aw * ah + bw * bh - inter
    return inter / area if area > 0 else 0.0


def _match_cached_result(box, cache, now):
    best = None; best_iou = 0.0
    for result in cache:
        if now - result["ts"] > RESULT_TTL_SEC:
            continue
        iou = _box_iou(box, result["box"])
        if iou > best_iou or (best is not None and iou == best_iou and result["ts"] > best["ts"]):
            best_iou = iou; best = result
            
    if best is not None and best_iou >= 0.10:
        cache.remove(best)
        return best
    return None


def _match_insightface_to_haar(if_faces, haar_boxes, scale):
    """Match InsightFace detections to Haar cascade boxes using IoU.

    Each InsightFace face is matched to the best-overlapping Haar box.
    A Haar box can only be claimed by one IF face (greedy, best-IoU-first).
    Returns list of (if_face_dict, haar_box_in_original_coords) tuples.
    """
    if not if_faces or not haar_boxes:
        return []

    # Build all (iou, if_idx, haar_idx) pairs
    pairs = []
    for i, fd in enumerate(if_faces):
        # fd["box"] is in the scaled frame coords; convert to original
        bx, by, bw, bh = fd["box"]
        if_box_orig = (int(bx / scale), int(by / scale),
                       int(bw / scale), int(bh / scale))
        for j, hbox in enumerate(haar_boxes):
            iou = _box_iou(if_box_orig, hbox)
            pairs.append((iou, i, j, if_box_orig))

    # Sort by IoU descending → greedy assignment
    pairs.sort(key=lambda t: t[0], reverse=True)
    used_if   = set()
    used_haar = set()
    matched   = []
    for iou_val, i, j, if_box_orig in pairs:
        if i in used_if or j in used_haar:
            continue
        if iou_val < 0.05:
            break  # remaining pairs have even lower IoU
        used_if.add(i)
        used_haar.add(j)
        matched.append((if_faces[i], haar_boxes[j]))

    # Any IF face that didn't match a Haar box — use its own coords
    for i, fd in enumerate(if_faces):
        if i not in used_if:
            bx, by, bw, bh = fd["box"]
            matched.append((fd, (int(bx / scale), int(by / scale),
                                 int(bw / scale), int(bh / scale))))
    return matched


def _recognition_worker(stop_event, tasks, result_cache, result_lock):
    """Background thread: pulls tasks and runs face recognition.

    Strategy:
    - When insightface backend is active: pass the full (scaled-down) frame
      to get_faces_with_embeddings() ONCE, then match results to Haar boxes.
      This is dramatically faster than calling get_embedding() per ROI because
      it avoids running the SCRFD detector N times.
    - When only ONNX/DNN backend: use the cropped ROI as before.
    """
    while not stop_event.is_set():
        try:
            task = tasks.get(timeout=0.1)
        except queue.Empty:
            continue

        start    = time.perf_counter()
        new_results: list[dict] = []
        scale    = task.get("scale", 1.0)
        idx_snap = task["index"]
        haar_boxes = task.get("boxes", [])

        # ── Path A: InsightFace full-frame (fast — one call for all faces) ───
        used_insightface = False
        if eng.has_insightface() and task.get("frame") is not None:
            faces_data = eng.get_faces_with_embeddings(task["frame"])
            if faces_data:
                used_insightface = True
                matched = _match_insightface_to_haar(faces_data, haar_boxes, scale)

                n_faces = len(matched)
                for face_idx, (fd, display_box) in enumerate(matched):
                    emb = fd["embedding"]
                    best, score, second, second_score = eng.find_best_match(emb, idx_snap)

                    margin = score - second_score
                    is_known = best is not None and margin >= 0.10

                    print(f"\n{'='*50}")
                    print(f"FRAME RECOGNITION (InsightFace)")
                    print(f"Detected faces: {n_faces}")
                    print(f"{'='*50}\n")
                    print(f"FACE #{face_idx + 1}")
                    bx, by, bw, bh = display_box
                    print(f"Bounding box: x={bx}, y={by}, w={bw}, h={bh}")
                    print(f"Candidates:")
                    print(f"    Best: {best['name'] if best else 'None'} ({score:.4f})")
                    print(f"    Second: {second['name'] if second else 'None'} ({second_score:.4f})")
                    print(f"Margin: {margin:.4f}")
                    print(f"Threshold: {eng.recognition_threshold():.4f}")
                    print(f"Decision: {'MATCH (' + best['name'] + ')' if is_known else 'UNKNOWN/REJECTED'}")
                    print("-" * 50)

                    new_results.append({
                        "box":   display_box,
                        "kind":  "known" if is_known else "unknown",
                        "entry": best if is_known else None,
                        "score": score,
                        "margin": margin,
                        "reason": "margin_too_low" if best and not is_known else ""
                    })

        # ── Path B: ONNX/DNN ROI fallback (no insightface or it found nothing)
        if not used_insightface and task.get("rois"):
            for roi, box in zip(task["rois"], task["boxes"]):
                emb = eng.get_embedding(roi)
                if emb is not None:
                    bx, by, bw, bh = box
                    best, score, second, second_score = eng.find_best_match(emb, idx_snap)

                    margin = score - second_score
                    is_known = best is not None and margin >= 0.10

                    print(f"\n{'='*50}")
                    print(f"FRAME RECOGNITION (ROI Fallback)")
                    print(f"Detected faces: {len(task['rois'])}")
                    print(f"{'='*50}\n")
                    print(f"FACE #{len(new_results) + 1}")
                    print(f"Bounding box: x={bx}, y={by}, w={bw}, h={bh}")
                    print(f"Candidates:")
                    print(f"    Best: {best['name'] if best else 'None'} ({score:.4f})")
                    print(f"    Second: {second['name'] if second else 'None'} ({second_score:.4f})")
                    print(f"Margin: {margin:.4f}")
                    print(f"Threshold: {eng.recognition_threshold():.4f}")
                    print(f"Decision: {'MATCH (' + best['name'] + ')' if is_known else 'UNKNOWN/REJECTED'}")
                    print("-" * 50)

                    new_results.append({
                        "box":   box,
                        "kind":  "known" if is_known else "unknown",
                        "entry": best if is_known else None,
                        "score": score,
                        "margin": margin,
                        "reason": "margin_too_low" if best and not is_known else ""
                    })

        elapsed_ms = (time.perf_counter() - start) * 1000
        completed_at = time.time()
        for result in new_results:
            result["ts"] = completed_at
        _set_perf(recognition_ms=round(elapsed_ms, 1))

        with result_lock:
            result_cache[:] = [
                r for r in result_cache
                if completed_at - r["ts"] <= RESULT_TTL_SEC
            ][-(MAX_RESULT_CACHE - 1):]
            result_cache.extend(new_results)


# ── Helper to replace newest item in queue (non-blocking) ─────────────────────

def _replace_latest(q: queue.Queue, item) -> None:
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_attendance(stop_event: threading.Event, use_flask: bool = False) -> None:
    global _output_frame
    if use_flask:
        with _frame_lock:
            _output_frame = _make_placeholder("Opening camera", "Please wait a moment")
    _set_camera_state("starting", "Opening camera")

    arcface_ok = eng.is_ready()
    if arcface_ok:
        reload_embeddings()
    else:
        st = eng.model_status()
        print(f"[ATT] ArcFace not ready: {st['error']}")

    cascade    = cv2.CascadeClassifier(CASCADE_PATH)
    cascade_ok = not cascade.empty()

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = _open_camera()
    if cap is None:
        print("[ATT] No camera found.")
        if use_flask:
            with _frame_lock:
                _output_frame = _make_placeholder(
                    "Camera unavailable", "Close other camera apps, then press Start"
                )
        _set_camera_state("error", "Camera unavailable")
        stop_event.set()
        return

    # ── Start dedicated camera-reader thread ──────────────────────────────────
    cam_buf      = [None]           # cam_buf[0] = latest frame
    cam_buf_lock = threading.Lock()
    cam_reader   = threading.Thread(
        target=_camera_reader,
        args=(cap, cam_buf, cam_buf_lock, stop_event),
        daemon=True,
    )
    cam_reader.start()

    # Wait up to 5 s for first frame
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with cam_buf_lock:
            if cam_buf[0] is not None:
                break
        time.sleep(0.05)

    with cam_buf_lock:
        first_frame_ready = cam_buf[0] is not None
    if not first_frame_ready:
        cap.release()
        cam_reader.join(timeout=0.8)
        if use_flask:
            with _frame_lock:
                _output_frame = _make_placeholder(
                    "Camera unavailable", "No video received. Close other camera apps, then retry"
                )
        _set_camera_state("error", "Camera opened but did not provide video")
        stop_event.set()
        return

    # ── State ─────────────────────────────────────────────────────────────────
    last_marked: dict[int, float]             = {}
    mark_states: dict[int, tuple[str, float]] = {}
    unk_timer        = 0.0
    proc_count       = 0       # frames processed (for DETECT_EVERY)
    last_submit      = 0.0
    last_encode      = 0.0     # for STREAM_FPS cap
    fps_count        = 0
    fps_window       = time.time()
    faces: tuple     = ()
    schedule_cfg     = load_time_cfg()
    next_cfg_refresh = time.time() + 10.0

    result_cache: list[dict] = []
    result_lock  = threading.Lock()
    tasks        = queue.Queue(maxsize=1)
    worker_thread = None
    if arcface_ok:
        worker_thread = threading.Thread(
            target=_recognition_worker,
            args=(stop_event, tasks, result_cache, result_lock),
            daemon=True,
        )
        worker_thread.start()

    st = eng.model_status()
    camera_failed = False

    while not stop_event.is_set():
        # ── Get latest frame (non-blocking) ───────────────────────────────────
        with cam_buf_lock:
            frame = cam_buf[0]

        if frame is None:
            time.sleep(0.01)
            continue

        frame = frame.copy()          # own copy so reader can overwrite buf safely
        proc_count += 1
        now = time.time()

        with _camera_lock:
            last_camera_frame = _camera_state["last_frame"]
        if last_camera_frame and now - last_camera_frame > CAMERA_STALL_SEC:
            print("[ATT] Camera stream stalled.")
            camera_failed = True
            _set_camera_state("error", "Camera stream stopped")
            if use_flask:
                with _frame_lock:
                    _output_frame = _make_placeholder(
                        "Camera stream stopped", "Press Start to reconnect"
                    )
            stop_event.set()
            break

        # ── FPS counter ───────────────────────────────────────────────────────
        fps_count += 1
        if now - fps_window >= 1.0:
            _set_perf(fps=round(fps_count / (now - fps_window), 1))
            fps_count = 0
            fps_window = now

        # ── Schedule config refresh ───────────────────────────────────────────
        if now >= next_cfg_refresh:
            schedule_cfg     = load_time_cfg()
            next_cfg_refresh = now + 10.0

        # ── ArcFace not ready — show error overlay ────────────────────────────
        if not arcface_ok:
            cv2.putText(frame, "ArcFace model not ready", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 240), 2)
            fix = ("Run: python download_models.py" if not st["files_exist"]
                   else "Run: python install.py")
            cv2.putText(frame, fix, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
            cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            _encode_and_store(frame, use_flask, now, last_encode)
            last_encode = now
            time.sleep(1.0 / STREAM_FPS)
            continue

        # ── Face detection (Haar cascade — lightweight, runs every Nth frame) ─
        if proc_count % DETECT_EVERY == 0 and cascade_ok:
            small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)
            raw   = cascade.detectMultiScale(
                gray, scaleFactor=1.12, minNeighbors=5,
                minSize=(MIN_FACE_PX, MIN_FACE_PX)
            )
            inv   = 1.0 / DETECT_SCALE
            faces = (
                tuple(
                    (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
                    for x, y, w, h in raw
                )
                if len(raw) else ()
            )

        # ── Submit recognition task ───────────────────────────────────────────
        with _embed_lock:
            idx_snap = list(_embed_idx)

        if faces and idx_snap and now - last_submit >= RECOGNIZE_INTERVAL_SEC:
            rois = []
            boxes = []
            for (x, y, w, h) in faces:
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                roi = eng.crop_face(frame, (x1, y1, x2 - x1, y2 - y1))
                rois.append(roi)
                boxes.append((x1, y1, x2 - x1, y2 - y1))

            # Build a half-resolution frame for insightface (much faster)
            recog_frame = None
            if eng.has_insightface() and frame.size > 0:
                recog_frame = cv2.resize(
                    frame, (0, 0),
                    fx=RECOG_FRAME_SCALE, fy=RECOG_FRAME_SCALE,
                    interpolation=cv2.INTER_AREA,
                )

            _replace_latest(tasks, {
                "frame": recog_frame,           # full (scaled) frame for insightface
                "rois":  rois,  # stable crops for ONNX/OpenCV fallback
                "boxes": boxes,
                "scale": RECOG_FRAME_SCALE if recog_frame is not None else 1.0,
                "index": idx_snap,
            })
            last_submit = now

        # ── Draw results ──────────────────────────────────────────────────────
        with result_lock:
            active_results = [
                r for r in result_cache
                if now - r["ts"] <= RESULT_TTL_SEC
            ]
            result_cache[:] = active_results[-MAX_RESULT_CACHE:]

        for (x, y, w, h) in faces:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            box    = (x1, y1, x2 - x1, y2 - y1)

            if not idx_snap:
                _brackets(frame, x1, y1, x2 - x1, y2 - y1, (0, 165, 220))
                _badge(frame, "No profiles", x1, y1, (0, 165, 220))
                continue

            result = _match_cached_result(box, active_results, now)

            if result is None:
                # No result yet — show scanning
                _brackets(frame, x1, y1, x2 - x1, y2 - y1, (0, 60, 200))
                _badge(frame, "Scanning", x1, y1, (0, 60, 200))
                continue

            if result["kind"] == "known" and result["entry"] is not None:
                user_entry = result["entry"]
                sim        = result["score"]
                uid        = user_entry["user_id"]
                name       = user_entry["name"]

                state, expire = mark_states.get(uid, ("none", 0.0))
                if now > expire: state = "none"
                col = (
                    (0, 255, 80)   if state == "marked"  else
                    (0, 200, 255)  if state == "already" else
                    (80, 255, 160)
                )
                _brackets(frame, x1, y1, x2 - x1, y2 - y1, col)
                _badge(frame, f"{name} {sim:.2f}", x1, y1, col)

                if now - last_marked.get(uid, 0.0) >= MARK_COOLDOWN_SEC:
                    arr        = time.strftime("%H:%M:%S")
                    att_status, lmin = classify_arrival(arr, schedule_cfg)
                    newly      = db.mark_attendance(uid, name, status=att_status,
                                                    late_minutes=lmin)
                    last_marked[uid] = now
                    if newly:
                        mark_states[uid] = ("marked", now + 4)
                        tag = f" (+{lmin}m)" if lmin > 0 else ""
                        bc  = (
                            (30, 200, 80)  if att_status == "on_time" else
                            (0, 165, 220)  if att_status == "late"    else
                            (30, 60, 200)
                        )
                        _banners.append((f"  Marked: {name}{tag}  ", now + 4, bc))
                        _push_event("marked", name, sim, att_status, lmin)
                        msg = f"Attendance marked. Welcome, {name}."
                        if lmin > 0: msg += f" {lmin} minutes late."
                        _speak(msg)
                        print(f"[ATT] Marked: {name} ({att_status}, sim={sim:.2f})")
                    else:
                        mark_states[uid] = ("already", now + 3)
                        _banners.append(
                            (f"  Already marked: {name}  ", now + 3, (0, 180, 220))
                        )
                        _push_event("already_marked", name, sim, att_status, lmin)
                        _speak(f"Already marked today, {name}.")

            elif result["kind"] == "unknown":
                sim = result["score"]
                reason = result.get("reason", "")
                label = f"Ambiguous {sim:.2f}" if reason == "margin_too_low" else f"Unknown {sim:.2f}"
                _brackets(frame, x1, y1, x2 - x1, y2 - y1, (0, 60, 200))
                _badge(frame, label, x1, y1, (0, 60, 200))
                if now - unk_timer >= MARK_COOLDOWN_SEC:
                    _push_event("unknown", "Unknown", sim)
                    _speak("Face not recognised.")
                    unk_timer = now

            else:
                # "processing" kind — still waiting on first result
                _brackets(frame, x1, y1, x2 - x1, y2 - y1, (0, 60, 200))
                _badge(frame, "Scanning", x1, y1, (0, 60, 200))

        _draw_banners(frame)
        cv2.putText(frame, time.strftime("%Y-%m-%d  %H:%M:%S"),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(frame, "ArcFace",
                    (frame.shape[1] - 82, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 255, 160), 1)

        # ── Encode & stream at capped STREAM_FPS ─────────────────────────────
        if use_flask:
            if now - last_encode >= 1.0 / STREAM_FPS:
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if ok:
                    with _frame_lock:
                        _output_frame = buf.tobytes()
                last_encode = now
        else:
            cv2.imshow("FaceAttend — ArcFace (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # No sleep here — we loop as fast as possible and let STREAM_FPS
        # cap control the encode rate.  Camera reader handles camera pacing.
        time.sleep(0.005)   # tiny yield so other threads get CPU time

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cam_reader.join(timeout=0.8)
    if worker_thread:
        worker_thread.join(timeout=0.5)
    if not camera_failed:
        with _frame_lock:
            _output_frame = None
        _set_camera_state("idle", "Camera inactive")
    if not use_flask:
        cv2.destroyAllWindows()
    print("[ATT] Camera released.")


def _encode_and_store(frame, use_flask, now, last_encode):
    if use_flask and now - last_encode >= 1.0 / STREAM_FPS:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with _frame_lock:
                global _output_frame
                _output_frame = buf.tobytes()


# ── Camera open ────────────────────────────────────────────────────────────────

def _open_camera() -> cv2.VideoCapture | None:
    """Open a working webcam with a Windows fallback if DirectShow is busy."""
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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

            # A backend may report success while another app still owns the
            # device. Confirm that it can actually deliver a frame before use.
            ready = False
            for _ in range(4):
                ret, _ = cap.read()
                if ret:
                    ready = True
                    break
            if ready:
                print(f"[ATT] Camera {idx} opened.")
                return cap
            cap.release()
    return None


# ── MJPEG generator ────────────────────────────────────────────────────────────

def _make_placeholder(title: str = "Camera off",
                      subtitle: str = "Press Start to begin") -> bytes:
    img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    img[:] = (25, 28, 36)
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    cv2.rectangle(img, (cx - 44, cy - 70), (cx + 44, cy + 18), (36, 40, 52), cv2.FILLED)
    cv2.rectangle(img, (cx - 44, cy - 70), (cx + 44, cy + 18), (70, 80, 100), 2)
    cv2.circle(img, (cx, cy - 26), 22, (90, 102, 124), 2)
    cv2.line(img, (cx - 56, cy - 82), (cx + 56, cy + 30), (70, 80, 100), 2)
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
    sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0]
    cv2.putText(img, title, (cx - title_size[0] // 2, cy + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (125, 135, 158), 2)
    cv2.putText(img, subtitle, (cx - sub_size[0] // 2, cy + 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (85, 94, 115), 1)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return buf.tobytes()


_PLACEHOLDER: bytes | None = None

def generate_frames(stop_event: threading.Event) -> Iterator[bytes]:
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        _PLACEHOLDER = _make_placeholder()
    while True:
        with _frame_lock:
            frame = _output_frame
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            if stop_event.is_set():
                return
            time.sleep(1 / STREAM_FPS)
            continue
        if stop_event.is_set():
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _PLACEHOLDER + b"\r\n"
            return
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _PLACEHOLDER + b"\r\n"
        time.sleep(0.35)


if __name__ == "__main__":
    s = threading.Event()
    run_attendance(s, use_flask=False)
