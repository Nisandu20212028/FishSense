# FishSense 🐟🛰️

**AI-Powered Fishing Zone Prediction System**

A machine learning-based system that uses satellite oceanographic data to identify potential fishing zones, helping small-scale fishermen and promoting sustainable fisheries management.

## 📋 Project Information

- **Student**: Nisandu Senanayake (w1871483)
- **Supervisor**: Kanishka Hewageegana
- **Degree**: BEng (Hons) Software Engineering
- **Institution**: University of Westminster
- **Module**: 6COSC023W - Computer Science Final Project

## 🎯 Project Objectives

FishSense aims to:
- Aggregate satellite-derived oceanographic data (SST, Chlorophyll-a, Ocean Currents)
- Apply machine learning algorithms (Random Forest, K-Means) to predict fishing hotspots
- Visualize predictions through an interactive web dashboard
- Support sustainable fisheries management

## 🛠️ Technologies Used

- **Python 3.10+**
- **Google Earth Engine** - Satellite data access
- **Scikit-learn** - Machine learning models
- **Streamlit** - Web dashboard
- **Matplotlib/Seaborn** - Data visualization
- **Pandas/NumPy** - Data processing

## 📁 Project Structure
```
FishSense/
├── data/              # Satellite data and samples
├── models/            # Trained ML models
├── scripts/           # Python scripts
├── dashboard/         # Streamlit web interface
├── notebooks/         # Jupyter notebooks for exploration
└── requirements.txt   # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Google Earth Engine account
- 16GB RAM recommended

### Installation

1. Clone the repository
```bash
git clone https://github.com/nisandu20212028/FishSense.git
cd FishSense
```

2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Authenticate Google Earth Engine
```bash
earthengine authenticate
```

### Usage

1. Explore available datasets:
```bash
python scripts/explore_datasets.py
```

2. Download sample data:
```bash
python scripts/download_sample_data.py
```

3. Visualize data:
```bash
python scripts/visualize_data.py
```

## 🌊 Study Area

**Sri Lankan Coastal Waters**
- Bounding Box: [79.5°E, 5.9°N, 81.9°E, 9.9°N]
- Area: ~117 km²

## 📊 Data Sources

1. **Sea Surface Temperature (SST)**
   - Source: NOAA CDR
   - Resolution: ~28 km
   - Update: Daily

2. **Chlorophyll-a Concentration**
   - Source: MODIS Aqua
   - Resolution: ~4 km
   - Update: Daily

3. **Ocean Currents**
   - Source: HYCOM
   - Resolution: ~9 km
   - Update: Daily

## 📈 Current Progress

- [x] Project setup and environment configuration
- [x] Google Earth Engine integration
- [x] Data exploration and visualization
- [ ] Data preprocessing pipeline
- [ ] Machine learning model development
- [ ] Web dashboard creation
- [ ] Model evaluation and testing

## 📝 License

This project is part of academic coursework at the University of Westminster.

## 👨‍💻 Author

**Nisandu Senanayake**
- University of Westminster
- w1871483

## 🙏 Acknowledgments

- Supervisor: Kanishka Hewageegana
- Google Earth Engine for satellite data access
- University of Westminster School of Computer Science & Engineering