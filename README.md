# Sign Language Recognition System

A real-time sign language recognition system that detects A-Z hand gestures using computer vision and machine learning.

## 🚀 Features

- **Complete A–Z Recognition**: Recognizes all 26 letters including dynamic J and Z gestures with spatial trajectory tracking
- **Real-time WebRTC Streaming**: Cloud-ready browser camera streaming with asynchronous landmark extraction
- **Interactive Two-Section Web App**:
  - **Live Sign Prediction**: Real-time sign prediction with HUD confidence, detected hand, and temporal smoothing
  - **ASL Visual Learning Guide (A–Z)**: Comprehensive reference guide with verified public domain handshape illustrations, anatomical positioning, thumb placement, palm orientation, and step-by-step movement instructions
- **Bilateral Hand Normalization**: Automatic left/right hand detection and parity mirroring — sign with your natural dominant hand (right or left)
- **Sequence-Based Motion Recognition**: Dynamic J (pinky hook) and Z (index zigzag stroke) trajectory verification
- **Temporal Prediction Smoothing**: Multi-frame consensus voting with explicit neutral resting posture and uncertain state filtering

## 🎯 Quick Start

### Streamlit Web App (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web app
streamlit run streamlit_app.py
```

### Desktop Application

```bash
# Run the desktop app
python app.py
```

## 📱 Deployment

### Streamlit Cloud (Free)

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path to `streamlit_app.py`
5. Deploy!

### Local Deployment

```bash
streamlit run streamlit_app.py --server.port 8501 --server.headless true
```

## 🔧 Usage

### Web Application Sections
1. **📹 Live Sign Prediction**:
   - Click **Enable Camera** to start live WebRTC video stream
   - View real-time predictions, confidence score, detected hand, and stability HUD
   - Click **Disable Camera** to stop stream
2. **📖 ASL Visual Learning Guide (A–Z)**:
   - Search by letter or keyword (e.g. "fist", "thumb", "hook")
   - Filter by classification (Static vs. Movement-Based)
   - Step through circular Previous/Next navigation
   - Review verified handshape diagrams and hand-usage guidance

### Supported Gestures

**Static Signs (A–I, K–Y)**: Hold hand position stable
- 24 static letter classes classified via TFLite keypoint neural network

**Dynamic Signs (J, Z)**: Trajectory-based spatial tracking
- **J**: Start in ASL 'I' (pinky upright) and trace a downward curving hook
- **Z**: Start in ASL '1' (index finger pointing) and draw a 3-stroke horizontal zigzag in the air

### Hand Usage Notes
- **Dominant Hand**: Sign with your dominant hand (right or left). The engine performs automatic bilateral normalization.
- **Lighting & Framing**: Keep your hand clearly visible within the camera frame at chest/shoulder level.

## 🏗️ System Architecture

The system uses a hybrid approach combining static gesture recognition with motion tracking:

1. **MediaPipe**: Hand landmark detection and tracking
2. **TensorFlow Lite**: Static gesture recognition (A-Y)
3. **Motion Analysis**: Dynamic gesture detection (J, Z)
4. **Streamlit**: Web interface and real-time processing

## 📊 Performance

- **Static Signs**: 71.12% accuracy on validation set
- **Real-time Processing**: Optimized for live detection
- **Motion Detection**: Simple and reliable J/Z recognition
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🛠️ Technical Details

### Dependencies
- Python 3.12 compatible
- MediaPipe for hand tracking
- TensorFlow Lite for model inference
- OpenCV for camera handling
- Streamlit for web interface

### Model Files
- `slr/model/slr_model.tflite`: Pre-trained TensorFlow Lite model
- `slr/model/label.csv`: Gesture labels and mappings

## 🎨 Customization

The system can be customized to recognize additional gestures:

1. Run `app.py` for data collection mode
2. Press `1` to switch to data collection
3. Make gestures and press corresponding letter keys
4. Train new model using `training.ipynb`

## 📁 Project Structure

```
Sign-Language-Recognition/
├── streamlit_app.py          # Main web application
├── app.py                    # Desktop application
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── slr/
│   ├── model/
│   │   ├── slr_model.tflite  # Trained model
│   │   └── label.csv         # Gesture labels
│   ├── main.py              # Desktop app entry point
│   ├── model/
│   │   └── classifier.py     # Model inference
│   └── utils/
│       └── pre_process.py    # Data preprocessing
└── README.md                # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This system is designed for educational and research purposes. For production use, consider additional testing and validation.