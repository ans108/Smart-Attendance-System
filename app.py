"""
app.py — Flask dashboard v8, ArcFace-only, all bugs fixed.
"""

import csv, io, json, os, queue, shutil, threading, time
from datetime import date, datetime

from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, session, url_for)

import database as db
import attendance as att
import face_register as fr
import admin_auth as auth
import recognition_engine as eng
from time_policy import (get_schedule_summary, load_config as load_time_cfg,
                         save_config, status_badge_class, status_label,
                         classify_arrival)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "faceattend-2026-secret-key")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ── Thread state ───────────────────────────────────────────────────────────────
_cam_lock       = threading.Lock()
_stop_event     = threading.Event(); _stop_event.set()
_recog_thread: threading.Thread | None = None

_reg_lock_app   = threading.Lock()
_reg_stop_event = threading.Event(); _reg_stop_event.set()
_reg_thread: threading.Thread | None = None


def _cam_worker(stop_event):
    att.run_attendance(stop_event, use_flask=True)

def _reg_worker(name, stop_event):
    label = fr.register_user_web(name, stop_event)
    if label > 0:
        att.reload_embeddings()
        print(f"[APP] Embeddings refreshed after registering '{name}'.")

# ── Pages ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           stats=db.get_today_stats(),
                           today=date.today().isoformat(),
                           schedule=get_schedule_summary(),
                           model_ready=eng.is_ready())

@app.route("/register")
def register_page():
    return render_template("register.html",
                           users=db.get_all_users(),
                           model_ready=eng.is_ready(),
                           model_status=eng.model_status())

@app.route("/records")
def records_page():
    fd=request.args.get("date",""); date_from=request.args.get("date_from","")
    date_to=request.args.get("date_to",""); nq=request.args.get("name","")
    records=db.get_attendance(filter_date=fd or None,date_from=date_from or None,
                              date_to=date_to or None,name_query=nq or None)
    return render_template("records.html",records=records,filter_date=fd,
                           date_from=date_from,date_to=date_to,name_query=nq,
                           schedule=get_schedule_summary(),
                           status_label=status_label,status_badge_class=status_badge_class)

# ── Admin pages ────────────────────────────────────────────────────────────────

@app.route("/admin")
def admin_root():
    return redirect(url_for("admin_dashboard") if auth.is_logged_in() else url_for("admin_login_page"))

@app.route("/admin/login", methods=["GET"])
def admin_login_page():
    if auth.is_logged_in(): return redirect(url_for("admin_dashboard"))
    error = session.pop("login_error", None)
    return render_template("admin_login.html", error=error)

@app.route("/admin/login", methods=["POST"])
def admin_login_post():
    u=request.form.get("username","").strip(); p=request.form.get("password","")
    if auth.check_credentials(u, p):
        auth.login_admin(); return redirect(url_for("admin_dashboard"))
    session["login_error"] = "Invalid username or password."
    return redirect(url_for("admin_login_page"))

@app.route("/admin/dashboard")
@auth.login_required
def admin_dashboard():
    st = eng.model_status()
    return render_template("admin.html",
                           overview=db.get_admin_overview(),
                           schedule=get_schedule_summary(),
                           students=db.get_student_analysis(),
                           weekly=db.get_weekly_chart(7),
                           emb_count=db.embedding_count(),
                           users=db.get_all_users(),
                           engine_ready=st["ready"],
                           engine_mode=st["mode"],
                           engine_status=st["error"] or "ArcFace buffalo_l — ready",
                           today=date.today().isoformat())

@app.route("/admin/logout")
def admin_logout():
    auth.logout_admin(); return redirect(url_for("admin_login_page"))

# ── Admin API ──────────────────────────────────────────────────────────────────

@app.route("/api/admin/overview")
@auth.login_required
def api_admin_overview(): return jsonify(db.get_admin_overview())

@app.route("/api/admin/weekly_chart")
@auth.login_required
def api_admin_weekly_chart():
    return jsonify(db.get_weekly_chart(int(request.args.get("days",7))))

@app.route("/api/admin/student_analysis")
@auth.login_required
def api_admin_student_analysis(): return jsonify(db.get_student_analysis())

@app.route("/api/admin/update_schedule", methods=["POST"])
@auth.login_required
def api_admin_update_schedule():
    data=request.get_json(force=True,silent=True) or {}
    allowed={"work_start","grace_minutes","work_end","work_end_ext","use_extended"}
    updates={k:v for k,v in data.items() if k in allowed}
    if not updates: return jsonify({"error":"No valid fields"}),400
    save_config(updates)
    return jsonify({"message":"Schedule updated","schedule":get_schedule_summary()})

@app.route("/api/admin/bulk_delete", methods=["POST"])
@auth.login_required
def api_admin_bulk_delete():
    data=request.get_json(force=True,silent=True) or {}
    ids=data.get("user_ids",[])
    if not ids: return jsonify({"error":"user_ids required"}),400
    deleted=[]
    for uid in ids:
        label=db.delete_user(int(uid))
        if label is not None:
            folder=os.path.join(DATASET_DIR,str(label))
            if os.path.isdir(folder): shutil.rmtree(folder)
            deleted.append(uid)
    att.reload_embeddings()
    return jsonify({"deleted":deleted,"count":len(deleted)})

@app.route("/api/admin/export_full")
@auth.login_required
def api_admin_export_full():
    rows=db.get_attendance()
    out=io.StringIO()
    w=csv.DictWriter(out,fieldnames=["id","name","date","time","status","late_minutes","created_at"],extrasaction="ignore")
    w.writeheader(); w.writerows(rows); out.seek(0)
    return Response(out.getvalue(),mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=attendance_full.csv"})


@app.route("/api/admin/manual_attendance", methods=["POST"])
@auth.login_required
def api_admin_manual_attendance():
    """Manually mark or update attendance for any user on any date."""
    data = request.get_json(force=True, silent=True) or {}
    user_id    = data.get("user_id")
    date_str   = (data.get("date") or "").strip()
    time_str   = (data.get("time") or "").strip()
    status     = (data.get("status") or "auto").strip()

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not date_str:
        return jsonify({"error": "date is required (YYYY-MM-DD)"}), 400
    if not time_str:
        return jsonify({"error": "time is required (HH:MM)"}), 400
    if status not in ("auto", "on_time", "late", "very_late"):
        return jsonify({"error": "Invalid attendance status."}), 400

    # Strictly validate values before they reach SQLite.  It keeps malformed
    # manual edits from producing records that charts and exports cannot read.
    try:
        manual_date = date.fromisoformat(date_str)
        if manual_date > date.today():
            raise ValueError("Future dates are not allowed")
        parsed_time = datetime.strptime(time_str, "%H:%M")
        time_str = parsed_time.strftime("%H:%M:%S")
        late_min = max(0, int(data.get("late_minutes") or 0))
    except ValueError as exc:
        return jsonify({"error": f"Invalid date, time, or late minutes: {exc}"}), 400

    if status == "auto":
        status, late_min = classify_arrival(time_str, load_time_cfg())
    elif status == "on_time":
        late_min = 0

    try:
        result = db.manual_mark_attendance(
            int(user_id), date_str, time_str, status, late_min
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Database error: {exc}"}), 500

    action = "created" if result["created"] else "updated"
    return jsonify({
        "message": f"Attendance {action} successfully.",
        "created": result["created"],
        "record":  result["record"],
    })

# ── Model status API ───────────────────────────────────────────────────────────

@app.route("/api/model_status")
def api_model_status():
    return jsonify(eng.model_status())

# ── Camera API ─────────────────────────────────────────────────────────────────

@app.route("/api/start")
def api_start():
    global _recog_thread, _stop_event
    with _cam_lock:
        if _reg_thread and _reg_thread.is_alive():
            return jsonify({"error": "Finish or cancel registration before starting the camera."}), 409
        if _recog_thread and _recog_thread.is_alive():
            if not _stop_event.is_set():
                return jsonify({"status":"already_running"})
            _recog_thread.join(timeout=2.0)
            if _recog_thread.is_alive():
                return jsonify({"error":"Camera is still closing. Please try again in a moment."}), 409
        _stop_event   = threading.Event()
        _recog_thread = threading.Thread(target=_cam_worker,args=(_stop_event,),daemon=True)
        _recog_thread.start()
    return jsonify({"status":"started", "message":"Opening camera"})

@app.route("/api/stop")
def api_stop():
    _stop_event.set()
    if _recog_thread and _recog_thread.is_alive():
        _recog_thread.join(timeout=2.0)
    return jsonify({"status":"stopped"})

@app.route("/api/recognition_status")
def api_recognition_status():
    running = bool(_recog_thread and _recog_thread.is_alive() and not _stop_event.is_set())
    st      = eng.model_status()
    perf    = att.runtime_status()
    return jsonify({
        "running": running,
        "model_ready": st["ready"],
        "engine": st["mode"],
        "fps": perf.get("fps", 0.0),
        "recognition_ms": perf.get("recognition_ms", 0.0),
        "camera_state": perf.get("state", "idle"),
        "camera_message": perf.get("message", "Camera inactive"),
    })

# ── Registration API ───────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def api_register():
    global _reg_thread, _reg_stop_event
    data=request.get_json(force=True,silent=True) or {}
    name=(data.get("name") or "").strip()
    if not name: return jsonify({"error":"Name is required"}),400
    if len(name) > 80:
        return jsonify({"error":"Name must be 80 characters or fewer."}),400
    user = db.get_user_by_name(name)
    if user:
        with db.get_connection() as conn:
            has_emb = conn.execute("SELECT 1 FROM embeddings WHERE user_id = ?", (user["id"],)).fetchone()
            if has_emb:
                return jsonify({"error":"A user with this name is already registered."}),409

    with _reg_lock_app:
        if _reg_thread and _reg_thread.is_alive():
            return jsonify({"error":"Registration already in progress"}),409
        _stop_event.set()
        if _recog_thread and _recog_thread.is_alive():
            _recog_thread.join(timeout=5.0)
        if _recog_thread and _recog_thread.is_alive():
            return jsonify({"error":"Camera is still closing. Try registration again in a moment."}),409
        _reg_stop_event = threading.Event()
        _reg_thread = threading.Thread(target=_reg_worker,args=(name,_reg_stop_event),daemon=True)
        _reg_thread.start()
    return jsonify({"message":f"Registration started for '{name}'."})

@app.route("/api/reg_progress")
def api_reg_progress(): return jsonify(fr.get_reg_status())

@app.route("/api/register_cancel", methods=["POST"])
def api_register_cancel():
    _reg_stop_event.set()
    if _reg_thread and _reg_thread.is_alive():
        _reg_thread.join(timeout=2.0)
    return jsonify({"status":"cancelled"})

@app.route("/reg_feed")
def reg_feed():
    return Response(fr.generate_reg_frames(_reg_stop_event),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Users API ──────────────────────────────────────────────────────────────────

@app.route("/api/users")
def api_users(): return jsonify(db.get_all_users())

@app.route("/api/delete_user", methods=["POST"])
def api_delete_user():
    data=request.get_json(force=True,silent=True) or {}
    uid=data.get("user_id")
    if not uid: return jsonify({"error":"user_id required"}),400
    label=db.delete_user(int(uid))
    if label is None: return jsonify({"error":"User not found"}),404
    folder=os.path.join(DATASET_DIR,str(label))
    if os.path.isdir(folder): shutil.rmtree(folder)
    att.reload_embeddings()
    return jsonify({"message":"User deleted"})

# ── SSE ────────────────────────────────────────────────────────────────────────

def _sse_stream():
    last_hb=time.time()
    try:
        while True:
            try:
                ev=att.event_queue.get(timeout=1.0)
                yield f"data: {json.dumps(ev)}\n\n"
            except queue.Empty:
                if time.time()-last_hb>=15:
                    yield ": heartbeat\n\n"; last_hb=time.time()
    except GeneratorExit: pass

@app.route("/api/events")
def api_events():
    return Response(_sse_stream(),mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Video feeds ────────────────────────────────────────────────────────────────

@app.route("/video_feed")
def video_feed():
    return Response(att.generate_frames(_stop_event),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Data API ───────────────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats(): return jsonify(db.get_today_stats())

@app.route("/api/attendance")
def api_attendance():
    return jsonify(db.get_attendance(
        filter_date=request.args.get("date"),date_from=request.args.get("date_from"),
        date_to=request.args.get("date_to"),name_query=request.args.get("name")))

@app.route("/download/csv")
def download_csv():
    fd=request.args.get("date"); df=request.args.get("date_from"); dt=request.args.get("date_to")
    rows=db.get_attendance(filter_date=fd,date_from=df,date_to=dt)
    out=io.StringIO()
    w=csv.DictWriter(out,fieldnames=["id","name","date","time","status","late_minutes","created_at"],extrasaction="ignore")
    w.writeheader(); w.writerows(rows); out.seek(0)
    lbl=fd or (f"{df}_to_{dt}" if df or dt else "all")
    return Response(out.getvalue(),mimetype="text/csv",
                    headers={"Content-Disposition":f"attachment; filename=attendance_{lbl}.csv"})

# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    st = eng.model_status()
    print()
    print("  +----------------------------------------------+")
    print("  |      FaceAttend - ArcFace Attendance         |")
    print("  +----------------------------------------------+")
    print("  |  Dashboard  -> http://127.0.0.1:5000         |")
    print("  |  Admin      -> http://127.0.0.1:5000/admin   |")
    print("  |  Login:  admin / admin123                    |")
    print("  +----------------------------------------------+")
    if st["ready"]:
        print("  |  ArcFace:  [OK] Ready                        |")
    else:
        print("  |  ArcFace:  [X] Model not loaded              |")
        if st["files_exist"]:
            print("  |  Run first:  python install.py               |")
        else:
            print("  |  Run first:  python download_models.py        |")
    print("  +----------------------------------------------+")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
