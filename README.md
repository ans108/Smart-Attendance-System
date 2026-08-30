# FaceAttend - Smart Attendance System

## 1. Project Overview
FaceAttend is a Python-based real-time face recognition and attendance tracking system. Its primary purpose is to automatically detect faces from a camera feed, identify registered users using state-of-the-art deep learning embedding models, and securely log their attendance (including arrival time and late status) into a central database, visible via a web dashboard.

---

## 2. Features
- **Face Registration**: A robust real-time user enrollment pipeline.
- **20-Sample Diversity Registration**: The system strictly captures 20 diverse, high-quality face samples per user, rejecting near-duplicate frames.
- **ArcFace 512-Dimensional Embeddings**: Extracts highly distinct 512-D facial feature vectors.
- **Prototype Embedding Generation**: Calculates a single representative, normalized spherical centroid from the 20 registered samples.
- **Single-Face & Multi-Face Recognition**: Accurately processes multiple faces in a single frame independently.
- **Unknown-Person Rejection**: Rejects unregistered faces that fall below the strict similarity threshold.
- **Attendance Tracking**: Automatically calculates 'On Time' vs 'Late' metrics.
- **Dashboard**: A live Flask-based web dashboard to view attendance events and system configuration.
- **SQLite Database**: Local, zero-configuration database for users, embeddings, and attendance logs.

---

## 3. Recognition Pipeline
The recognition process operates entirely locally and asynchronously to maintain a high-FPS camera feed:

    Camera Frame
        ↓
    Haar Cascade Face Detection
        ↓
    Face Alignment & Bounding Box Extraction
        ↓
    ArcFace Embedding (w600k_r50.onnx)
        ↓
    512-D Normalized Embedding
        ↓
    Cosine Similarity Comparison With User Prototypes
        ↓
    Threshold (0.5500) & Match Margin Evaluation
        ↓
    Identity / Unknown Decision
        ↓
    Attendance Logging & Caching

---

## 4. Model Information
The system relies on a hybrid pipeline optimized for speed and accuracy:
- **Model Pack**: `buffalo_l`
- **Detection Model**: OpenCV Haar Cascades (`haarcascade_frontalface_default.xml`) is used for extreme low-latency Region-of-Interest (ROI) face detection.
- **Recognition Model**: ArcFace (`w600k_r50.onnx`).
- **Embedding Dimension**: 512-D.
- **Runtime/Backend**: `onnxruntime` (Direct execution on CPU/GPU depending on environment). 

---

## 5. Face Registration
The registration process requires capturing exactly **20 face samples**. 
During registration, the system:
1. Detects the face and extracts a 512-D ArcFace embedding in real-time.
2. Checks the similarity against all previously captured samples in the session. If the similarity is `> 0.97`, the frame is rejected to enforce diversity (requiring the user to slightly turn their head).
3. Once 20 diverse samples are collected, a **Representative Prototype** is generated. This is computed as the normalized spherical centroid (mean embedding) of the 20 samples.
4. The single Prototype embedding is evaluated for cross-user duplication and stored in the database.

---

## 6. Recognition
Live recognition triggers whenever a face is detected in the frame:
- **Cosine Similarity**: The live embedding is compared against all registered prototypes using Cosine Similarity.
- **Threshold**: A strict recognition threshold of **`0.55`** is applied.
- **Match Margin**: If two registered identities match the live face with scores above the threshold, the system requires a safe margin between the highest and second-highest score to prevent ambiguity.
- **Result Caching**: Successful recognition results are cached spatially based on Bounding Box Intersection-over-Union (IoU) for up to 3 seconds to throttle redundant database writes.

---

## 7. Multi-Face Recognition
The pipeline is designed to support multiple faces simultaneously in the same frame.
- Every detected ROI box is sent to the ArcFace model independently.
- e.g., `Face #1 → Embedding #1 → Match #1`, `Face #2 → Embedding #2 → Match #2`.
- Cached matches strictly enforce identity exclusivity; a single cached identity cannot be assigned to multiple bounding boxes in the same frame.

---

## 8. Unknown Face Handling
Unregistered individuals are processed normally but ultimately rejected:
- If the best cosine similarity score against all database prototypes falls below the `0.55` threshold, the face is explicitly classified as **Unknown**.
- Unknown faces are briefly shown in the video feed but do not trigger an attendance database write.

---

## 9. Attendance System
- **Event Logging**: When a registered face is confidently matched, the system records the `user_id`, `name`, `date`, and `time` into the SQLite database.
- **Time Policy**: Configurable thresholds dictate whether the arrival is marked as `on_time` or `late`.
- **Throttling**: To prevent database spam, subsequent detections of the same user on the same day are recorded as `already_marked` in the dashboard but do not duplicate the database entry.

---

## 10. Project Structure
The project is strictly contained to required production files:
```text
smart_attendance/
├── app.py                  # Main Flask Web Server
├── attendance.py           # Background Recognition & Camera Engine
├── face_register.py        # 20-Sample Enrollment Logic
├── recognition_engine.py   # Embedding Generation & ONNX Runtime
├── database.py             # SQLite Operations
├── admin_auth.py           # Dashboard Authentication
├── time_policy.py          # Attendance Time/Late Rules
├── download_models.py      # ArcFace ONNX Downloader
├── install.py              # Environment Setup
├── requirements.txt        # Python Dependencies
├── attendance.db           # SQLite Production Database
├── models/
│   └── buffalo_l/          # InsightFace ONNX Models (w600k_r50.onnx)
├── static/                 # UI Assets (CSS/JS)
├── templates/              # HTML Templates
└── README.md
```

---

## 11. Installation
1. Ensure you have **Python 3.8+** installed.
2. (Optional but Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the required ArcFace models:
   ```bash
   python download_models.py
   ```

---

## 12. Running the Application
The entire application runs from a single entry point:
```bash
python app.py
```
- **Dashboard**: Access the UI at `http://127.0.0.1:5000`
- **Admin Panel / Registration**: Access at `http://127.0.0.1:5000/admin` (Default login: `admin` / `admin123`)
- From the admin panel, you can click "Register New User" to enroll a person.
- The camera will automatically start processing frames for live recognition.

---

## 13. Database
The system uses an embedded SQLite database (`attendance.db`) containing three primary tables:
- **`users`**: Stores the registered user's `id`, `name`, and unique integer `label`.
- **`embeddings`**: Stores the user's `user_id` and the 512-D prototype `embedding` (BLOB).
- **`attendance`**: Stores historical logs linking the `user_id` to the `date`, `time`, and `status`.

---

## 14. Testing
For automated evaluation of the recognition engine, you can isolate the pipeline and inject static images directly to `recognition_engine.get_embedding(roi)` and test matches using `compute_sim()`. The system supports robust testing for:
- Single face accuracy.
- Multi-face overlapping boxes.
- Registered vs Unknown disambiguation.

---

## 15. Troubleshooting
- **`onnxruntime` Import Failed**: If running on Windows, ensure `msvc-runtime` is installed (handled by `requirements.txt`).
- **No Face Detected During Registration**: Ensure good lighting and look directly into the camera. The Haar Cascade requires clear frontal features to extract the initial ROI.
- **Model Download Fails**: Ensure you have an active internet connection or manually place `w600k_r50.onnx` into `models/buffalo_l/`.

---

## 16. Performance
Based on local benchmarks:
- **Embedding Generation**: < 100ms per face using `onnxruntime` CPU execution.
- **Matching**: < 1ms across hundreds of registered prototypes.
- The system easily handles multi-face real-time streams at 15-30 FPS.

---

## 17. Security / Data Notes
- **Privacy**: Raw face images are **never** saved to disk during registration or recognition.
- **Embeddings**: Only mathematical representations (512-D float arrays) are stored in the database. These cannot be reverse-engineered into the original image.
- **Backup**: Always backup `attendance.db` before upgrading the environment.
