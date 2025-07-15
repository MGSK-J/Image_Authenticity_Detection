# Image Tampering Detection Web Application

A Flask-based web application for detecting image tampering using multiple machine learning models.

## Features

- **Multiple Detection Methods**: ELA, Histogram Equalization, Laplacian Edge, Noise Extraction, Pixel Analysis, and Hybrid Ensemble
- **Web Interface**: User-friendly drag-and-drop file upload
- **REST API**: JSON API endpoint for programmatic access
- **Real-time Analysis**: Instant results with confidence scores
- **Responsive Design**: Works on desktop and mobile devices

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Models**:
   - Ensure all model files are in the `models/` folder:
     - `ela_model.h5`
     - `histogram_equalization_model.h5`
     - `laplacian_edge_model.h5`
     - `noise_model.h5`
     - `pixel_analysis_model.h5`
     - `hybrid_ensemble_model.h5`

3. **Run Application**:
   ```bash
   python app.py
   ```

4. **Access Application**:
   - Web Interface: http://localhost:5000
   - API Endpoint: POST to http://localhost:5000/api/predict

## Usage

### Web Interface
1. Open http://localhost:5000 in your browser
2. Drag and drop an image or click to browse
3. Click "Analyze Image" to get results
4. View detailed analysis from all models

### API Usage
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```

## Supported Formats
- PNG, JPG, JPEG, GIF, BMP, TIFF
- Maximum file size: 16MB

## Model Information
- **ELA Model**: Error Level Analysis detection
- **Histogram Model**: Histogram equalization-based detection  
- **Laplacian Model**: Edge detection analysis
- **Noise Model**: Advanced noise extraction analysis
- **Pixel Model**: Pixel-level manipulation detection
- **Hybrid Ensemble**: Combined model for highest accuracy
