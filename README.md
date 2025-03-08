# facemaker-fm2
A comprehensive toolkit for hardware-accelerated image/video face swapping using ONNX framework.

## Features

- 🖼️ **Image Processing**:
  - Face Swapping
  - Face Enhancement
  - Face Detection
- 🎥 **Video Processing**:
  - Frame-by-frame Face Swapping
  - Face Enhancement
  - Face Detection & Tracking
- ✨ **Advanced Capabilities**:
  - Multi-face Detection
  - Face Direction Control
  - Face Enhancement Models
  - Precision Control (FP32/FP16)
- 🌐 **Multiple Interfaces**:
  - Web UI (Gradio)
  - REST API (Flask)
  - Python Library

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ikmalsaid/facemaker-fm2.git
cd facemaker-fm2
```

2. Create a Python virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Python Library

```python
from main import FacemakerFM2

# Initialize
fm = FacemakerFM2()

# Image face swapping
result = fm.recognize_from_image(
    target_paths="path/to/target.jpg",
    source_paths="path/to/source.jpg",
    source_face_index=0,
    target_face_index=0,
    swap_all_faces=False,
    detect_direction='left-right'
)

# Video face swapping
result = fm.recognize_from_video(
    target_paths="path/to/video.mp4",
    source_paths="path/to/source.jpg",
    source_face_index=0,
    target_face_index=0,
    swap_all_faces=True,
    detect_direction='left-right'
)

# Image face detection
faces, count, thumbnails, marked = fm.get_faces_from_image(
    "path/to/image.jpg",
    thumbnail=True,
    return_marked_image=True,
    detect_direction='left-right',
    unique_faces=True,
    similarity_threshold=0.4
)

# Video face detection
faces, count, thumbnails, marked = fm.get_faces_from_video(
    "path/to/video.mp4",
    thumbnail=True,
    return_marked_video=True,
    detect_direction='left-right',
    unique_faces=True,
    sample_rate=10,
    similarity_threshold=0.8
)

# Image face enhancement
result = fm.enhance_from_image(
    target_paths="path/to/image.jpg",
    target_face_index=0,
    enhance_all_faces=True,
    detect_direction='left-right'
)

# Video face enhancement
result = fm.enhance_from_video(
    target_paths="path/to/video.mp4",
    target_face_index=0,
    enhance_all_faces=True,
    detect_direction='left-right'
)
```

### Web UI

Start the Gradio web interface:

```python
fm = FacemakerFM2(mode='webui')
# OR
fm.start_webui(
    host="0.0.0.0",
    port=3225,
    browser=True,
    upload_size="10MB",
    public=False,
    limit=10
)
```

### REST API

Start the Flask API server:

```python
fm = FacemakerFM2(mode='api')
# OR
fm.start_api(
    host="0.0.0.0",
    port=3223,
    debug=False
)
```

#### API Endpoints

- `POST /api/swap/image`: Swap faces in image
- `POST /api/swap/video`: Swap faces in video
- `POST /api/enhance/image`: Enhance faces in image
- `POST /api/enhance/video`: Enhance faces in video
- `POST /api/detect/image`: Detect faces in image
- `POST /api/detect/video`: Detect faces in video
- `POST /api/settings`: Update processing settings
- `GET /api/download/<filename>`: Download processed files

## Configuration

```python
fm.change_settings(
    # Model Precision
    face_swapper_precision='fp32',  # 'fp32' or 'fp16'
    
    # Face Detection Models
    face_landmarker='peppa_wutz',   # '2dfan4' or 'peppa_wutz'
    face_enhancer='gfpgan_1.4',     # 'gfpgan_1.4', 'gfpgan_1.3', 'restoreformer', 'codeformer', 'gpen_bfr_512'
    
    # Detection Parameters
    face_detector_score=0.799,       # Confidence threshold (0.0 to 1.0)
    reference_face_distance=0.5,     # Face similarity threshold (0.0 to 1.0)
    
    # Output Settings
    save_format='webp',             # 'webp', 'jpg', or 'png'
    save_to='/path/to/output',      # Output directory path
    
    # Processing Options
    skip_enhancer=False             # Skip face enhancement step
)
```

### Face Detection Directions

The `detect_direction` parameter can be set to any of these values:
- `'left-right'`: Process faces from left to right
- `'right-left'`: Process faces from right to left
- `'top-bottom'`: Process faces from top to bottom
- `'bottom-top'`: Process faces from bottom to top
- `'small-large'`: Process faces from smallest to largest
- `'large-small'`: Process faces from largest to smallest
- `'best-worst'`: Process faces from highest to lowest quality
- `'worst-best'`: Process faces from lowest to highest quality

## License

See [LICENSE](LICENSE) for details.
