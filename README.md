# TLC Plate Analyzer 🧪

Automated TLC (Thin Layer Chromatography) analysis using deep learning.

## 🎯 Features

- **Automatic Spot Detection** - YOLOv8-based spot detection
- **Rf Value Calculation** - Precise retention factor computation
- **Solvent System Recommendation** - AI-powered solvent selection
- **Column Prediction** - Estimates number of columns needed
- **Time Estimation** - Predicts chromatography runtime

## 📊 Model Performance

- **Spot Detection Accuracy**: 100%
- **Solvent Classification**: 95%
- **Inference Speed**: ~7ms per image

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/TLC-Analyzer.git
cd TLC-Analyzer

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage
```python
from src.inference import TLCAnalyzer

# Initialize analyzer
analyzer = TLCAnalyzer(
    yolo_model_path='models/yolo_spot_detector.pt',
    recommendation_model_path='models/recommendation_model.pth'
)

# Analyze TLC image
result = analyzer.analyze_tlc('your_tlc_image.jpg')
print(result)

# Create visualization
analyzer.analyze_and_visualize('your_tlc_image.jpg', 'output.jpg')
```

## 📦 Project Structure
```
TLC_Analyzer_Project/
├── models/                      # Trained models
│   ├── yolo_spot_detector.pt
│   └── recommendation_model.pth
├── src/                         # Source code
│   └── inference.py
├── data/                        # Sample TLC images
├── sample_results/              # Example outputs
├── notebooks/                   # Training notebooks
├── requirements.txt
└── README.md
```

## 🧠 Models

### 1. YOLO Spot Detector
- **Architecture**: YOLOv8n
- **Training Data**: 500 synthetic TLC images
- **Performance**: mAP50 = 0.95+

### 2. Recommendation Model
- **Architecture**: Multi-task Neural Network
- **Tasks**: 
  - Solvent classification (3 classes)
  - Column count prediction (regression)
  - Time estimation (regression)

## 📈 Results

| Metric | Value |
|--------|-------|
| Spot Detection Accuracy | 100% |
| Solvent Classification | 95% |
| Columns MAE | 0.507 |
| Time MAE | 54.4 min |

## 🔧 Technical Details

- **Framework**: PyTorch
- **Object Detection**: Ultralytics YOLOv8
- **Image Processing**: OpenCV
- **Training**: Google Colab (GPU)

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

## 👨‍💻 Author

Your Name - [Your GitHub Profile](https://github.com/YOUR_USERNAME)

## 🙏 Acknowledgments

- Ultralytics YOLOv8
- PyTorch Team
- OpenCV Community

---

**Note**: This project uses synthetic training data. For production use, fine-tune with real TLC images.
