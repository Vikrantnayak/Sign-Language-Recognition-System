# Streamlit Community Cloud & Deployment Guide

## 1. Cloud Architecture Overview

In earlier versions, the web app called server-side `cv2.VideoCapture(0)`, which fails on containerized cloud platforms (such as Streamlit Community Cloud) because headless containers have no `/dev/video0` hardware device.

The platform now uses **`streamlit-webrtc`** with STUN server configuration:
- The user's browser opens their camera directly via HTML5 `getUserMedia`.
- Video frames are transmitted over an encrypted WebRTC peer connection to the application.
- Processing (MediaPipe Hand Landmark extraction + TFLite inference) executes asynchronously in Python background worker threads.
- Works seamlessly on both **Streamlit Community Cloud** and **Localhost** without local camera device locks.

---

## 2. Deploying to Streamlit Community Cloud

1. **Commit and Push to GitHub**:
   Ensure all files under `Sign-Language-Recognition-master` are committed.
2. **Open Streamlit Cloud Console**:
   Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. **Configure New App**:
   - **Repository**: Your GitHub repository
   - **Branch**: `main`
   - **Main file path**: `Sign-Language-Recognition-master/streamlit_app.py` (or `streamlit_app.py` if deployed from the subfolder root)
4. **Python Version**: Select **Python 3.12** (or 3.11).
5. **Click "Deploy"**.

---

## 3. Deployment Configuration & Files

| File | Purpose |
| :--- | :--- |
| `streamlit_app.py` | Web application entry point (WebRTC live streaming + ASL Visual Learning Guide) |
| `requirements.txt` | Lean cloud runtime dependencies (purged of desktop-only packages like `pyautogui`) |
| `requirements-desktop.txt` | Extended dependencies for Desktop OpenCV window and model training |
| `packages.txt` | OS-level system libraries (`libgl1`, `libglib2.0-0`) required for headless image handling |
| `.streamlit/config.toml` | Streamlit server configuration (headless mode, dark theme, disabled usage tracking) |
| `slr/paths.py` | Centralized relative path resolution ensuring all assets resolve regardless of working directory |
| `slr/model/` | Trained TFLite models (`slr_model.tflite`) and label maps (`label.csv`) |

---

## 4. Local Execution

### Web Application (Streamlit)
```bash
# Install cloud dependencies
pip install -r requirements.txt

# Launch web app
streamlit run streamlit_app.py
```

### Desktop Application (OpenCV GUI)
```bash
# Install full desktop dependencies (including pyautogui, scikit-learn, matplotlib)
pip install -r requirements-desktop.txt

# Run desktop application
python app.py
```

---

## 5. Automated Verification
Run the automated unit test suite from the project directory:
```bash
python -m unittest discover tests -v
```
