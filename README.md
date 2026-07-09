<div align="center">

<img src="docs/Tlc banner.png" alt="TLC Plate Analyzer Overview" width="100%"/>

# 🧪 TLC Plate Analyzer

### Automated TLC Analysis with Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="License">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FFD43B?style=for-the-badge" alt="YOLOv8">
</p>

**Detect spots • Calculate Rf values • Recommend solvents • Estimate time**

<p align="center">
  <a href="#-features"><strong>Features</strong></a> ·
  <a href="#-model-performance"><strong>Performance</strong></a> ·
  <a href="#-installation"><strong>Installation</strong></a> ·
  <a href="#-usage"><strong>Usage</strong></a> ·
  <a href="#-models"><strong>Models</strong></a> ·
  <a href="#-results"><strong>Results</strong></a>
</p>

</div>

---

Automated **TLC (Thin Layer Chromatography)** analysis using deep learning — upload a plate image and get spot detection, Rf values, solvent recommendations, and runtime estimates in one pass.

---

## 🎯 Features

| | |
|---|---|
| 🔍 **Automatic Spot Detection** | YOLOv8-based spot detection on TLC plate images |
| 📐 **Rf Value Calculation** | Precise retention factor computation for every detected spot |
| 🧪 **Solvent System Recommendation** | AI-powered solvent selection based on plate characteristics |
| 📊 **Column Prediction** | Estimates the number of columns needed for separation |
| ⏱️ **Time Estimation** | Predicts chromatography runtime |

---

## 📊 Model Performance

<div align="center">

| Metric | Value |
|---|---|
| 🎯 Spot Detection Accuracy | **100%** |
| 🧪 Solvent Classification | **95%** |
| ⚡ Inference Speed | **~7ms** per image |

</div>

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Safi-ullah-majid/TLC-Analyzer.git
cd TLC-Analyzer

# Install dependencies
pip install -r requirements.txt
```

---

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

---

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

---

## 🧠 Models

<table>
<tr>
<td width="50%" valign="top">

### 1️⃣ YOLO Spot Detector
- **Architecture**: YOLOv8n
- **Training Data**: 500 synthetic TLC images
- **Performance**: mAP50 = 0.95+

</td>
<td width="50%" valign="top">

### 2️⃣ Recommendation Model
- **Architecture**: Multi-task Neural Network
- **Tasks**:
  - Solvent classification (3 classes)
  - Column count prediction (regression)
  - Time estimation (regression)

</td>
</tr>
</table>

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Spot Detection Accuracy | 100% |
| Solvent Classification | 95% |
| Columns MAE | 0.507 |
| Time MAE | 54.4 min |

---

## 🔧 Technical Details

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Object%20Detection-YOLOv8-FFD43B?style=flat-square" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Image%20Processing-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Training-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" alt="Colab">
</p>

- **Framework**: PyTorch
- **Object Detection**: Ultralytics YOLOv8
- **Image Processing**: OpenCV
- **Training**: Google Colab (GPU)

---

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Safi Ullah Majid**

- GitHub: [@Safi-ullah-majid](https://github.com/Safi-ullah-majid)

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch Team](https://pytorch.org/)
- [OpenCV Community](https://opencv.org/)

---

<div align="center">

> ⚠️ **Note**: This project uses synthetic training data. For production use, fine-tune with real TLC images.

**⭐ Star this repository if you find it helpful!**

</div>
