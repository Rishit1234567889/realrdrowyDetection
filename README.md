# Driver Drowsiness Detection System

A comprehensive, Docker-based real-time driver drowsiness detection system using computer vision and machine learning. The system monitors driver alertness through Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head tilt detection, providing real-time alerts and analytics through a web dashboard.

## 🚗 Features

### Core Detection Capabilities
- **Eye Aspect Ratio (EAR)** monitoring for drowsy eyes detection
- **Mouth Aspect Ratio (MAR)** analysis for yawn detection  
- **Head Tilt** measurement using facial landmarks
- **Real-time video processing** with live camera feed
- **Multi-threshold alerting** system with consecutive frame validation

### Web Dashboard
- **Live MJPEG video stream** with annotated detection overlays
- **Real-time metrics display** with dynamic charts (Chart.js)u
- **Alert history tracking** with detailed logs
- **Responsive design** with Bootstrap styling
- **Connection status monitoring**

### API Endpoints
- `GET /` - Main dashboard interface
- `GET /video_feed` - MJPEG video stream
- `GET /metrics` - Current detection metrics (JSON)
- `GET /metrics/history` - Historical data for charting
- `GET /alerts` - Alert history
- `POST /alerts` - Log new alerts
- `GET /api/docs` - API documentation

## 🔧 Technical Specifications

### Detection Pipeline
- **Face Detection**: Haar Cascade (fast) + Dlib 68-landmark predictor (precise)
- **EAR Calculation**: `(|p2-p6| + |p3-p5|) / (2 * |p1-p4|)`
- **MAR Calculation**: `(|p14-p18| + |p15-p17|) / (2 * |p12-p16|)`
- **Head Tilt**: Angle calculation using nose tip and chin landmarks

### Thresholds & Parameters
- **EAR Threshold**: 0.25 (configurable)
- **MAR Threshold**: 0.6 (configurable)  
- **Head Tilt Threshold**: ±15° (configurable)
- **Consecutive Frames**: 20 frames for alert trigger
- **Video Resolution**: 640x480 @ 30 FPS

### Tech Stack
- **Backend**: Python 3.8+, Flask, OpenCV, Dlib
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js
- **Computer Vision**: OpenCV 4.8+, Dlib 19.24+, NumPy, SciPy
- **Containerization**: Docker, Docker Compose

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Webcam connected and accessible
- Linux/macOS (Windows requires additional camera configuration)

### Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd driver-drowsiness-detection
```

2. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

3. **Access the dashboard**
- Dashboard: http://localhost:5000/dashboard
- Video Feed: http://localhost:5000/video_feed  
- API Docs: http://localhost:5000/api/docs

### Manual Installation (Alternative)

1. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

2. **Download dlib face landmarks**
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

3. **Run the application**
```bash
python app.py
```

## 🐳 Docker Configuration

### Camera Access Setup

**Linux:**
```bash
# Check camera device
ls /dev/video*

# Run with camera access
docker run --device=/dev/video0:/dev/video0 -p 5000:5000 drowsiness-detector
```

**Windows (WSL2):**
```bash
# Install usbipd for USB device sharing
# Connect camera through WSL2
usbipd wsl attach --busid <bus-id>
```

**macOS:**
```bash
# Camera access requires additional configuration
# May need to run without Docker for camera access
python app.py
```

### Environment Variables
```bash
export FLASK_ENV=production
export CAMERA_INDEX=0  # Camera device index
export EAR_THRESHOLD=0.25
export MAR_THRESHOLD=0.6
export HEAD_TILT_THRESHOLD=15.0
```

## 📊 Dashboard Interface

### Real-time Metrics Cards
- **EAR Value**: Current eye aspect ratio with trend
- **MAR Value**: Current mouth aspect ratio  
- **Head Tilt**: Current head angle in degrees
- **Alert Status**: Normal/Drowsy/Yawning/Head Tilted

### Live Video Feed
- Real-time camera stream with detection overlays
- Facial landmarks visualization
- Current metrics displayed on video

### Interactive Charts
- Time-series plotting of EAR, MAR, and head tilt
- Dual y-axis for different metric scales
- Configurable time window (last 50 data points)

### Alert History Table
- Timestamp and alert type logging
- Detailed metrics at time of alert
- Exportable data format

## 🔬 API Usage Examples

### Get Current Metrics
```bash
curl http://localhost:5000/metrics
```
Response:
```json
{
  "ear": 0.285,
  "mar": 0.421,
  "head_tilt": 8.2,
  "alert_status": "Normal",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

### Log Custom Alert
```bash
curl -X POST http://localhost:5000/alerts \
  -H "Content-Type: application/json" \
  -d '{"alert_type": "Custom Alert", "severity": "high"}'
```

### Get Historical Data
```bash
curl http://localhost:5000/metrics/history
```

## ⚙️ Configuration & Tuning

### Detection Thresholds
Edit the following variables in `app.py`:
```python
EAR_THRESHOLD = 0.25      # Lower = more sensitive to eye closure
MAR_THRESHOLD = 0.6       # Higher = more sensitive to yawning  
HEAD_TILT_THRESHOLD = 15.0  # Degrees from vertical
CONSECUTIVE_FRAMES = 20    # Frames before alert trigger
```

### Camera Settings
```python
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
```

### Performance Optimization
- Reduce frame size for faster processing
- Adjust detection frequency
- Use hardware acceleration if available
- Implement frame skipping for lower-end hardware

## 🔧 Troubleshooting

### Common Issues

**Camera Not Found:**
```bash
# Check available cameras
ls /dev/video*
# Or on Windows
Device Manager > Cameras
```

**Docker Camera Access:**
```bash
# Add user to video group (Linux)
sudo usermod -a -G video $USER

# Check camera permissions
sudo chmod 666 /dev/video0
```

**Performance Issues:**
- Reduce video resolution in camera settings
- Increase CONSECUTIVE_FRAMES threshold
- Use fewer facial landmarks for detection

**Missing Dependencies:**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# Install dlib with cmake
pip install cmake dlib
```

### Debug Mode
```bash
# Run with debug logging
export FLASK_ENV=development
python app.py
```

## 📈 Performance Metrics

### System Requirements
- **CPU**: 2+ cores, 2.0 GHz minimum
- **RAM**: 1GB minimum, 2GB recommended
- **Camera**: USB 2.0+ webcam, 480p minimum
- **Network**: For dashboard access only

### Benchmarks
- **Detection Speed**: ~15-30 FPS (depending on hardware)
- **Latency**: <100ms end-to-end
- **Memory Usage**: ~300-500MB
- **CPU Usage**: 20-40% on modern hardware

## 🛠️ Development

### Project Structure
```
driver-drowsiness-detection/
├── app.py                 # Main Flask application
├── templates/
│   └── dashboard.html     # Web dashboard
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── README.md            # Documentation
```

### Adding Features
1. **New Detection Algorithms**: Extend `DrowsinessDetector` class
2. **Additional Metrics**: Add endpoints in Flask app
3. **Dashboard Enhancements**: Modify `dashboard.html`
4. **Alert Integrations**: Extend alert logging system

### Testing
```bash
# Unit tests for detection algorithms
python -m pytest tests/

# API endpoint testing
curl -X GET http://localhost:5000/metrics
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review API documentation at `/api/docs`

## 🔮 Future Enhancements

- [ ] Machine learning model integration (CNN-based detection)
- [ ] Mobile app companion
- [ ] Cloud deployment options
- [ ] Advanced alert notifications (SMS, email)
- [ ] Multi-camera support
- [ ] Driver identification and personalization
- [ ] Integration with vehicle systems
- [ ] Real-time data analytics and reporting
