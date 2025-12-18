Football Player Market Value Prediction
"Which model can predict better the value of a player in the FIFA 21 dataset?"
This project aims to predict professional football players’ market values using
data from the FIFA 21 dataset. 
It combines exploratory data analysis, feature engineering, and machine learning
models to estimate player value and analyze the key drivers behind it.

\*\*\*PROJECT STRUCTURE
Projet-foot/
├── main.py # Main pipeline entry point
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment
├── README.md
│
├── src/
│ ├── data\_loader.py # Data loading functions
│ ├── features.py # Feature engineering
│ ├── eda.py # Exploratory Data Analysis (EDA)
│ ├── modeling.py # Preprocessing and model definitions
│ ├── training.py # Training and model selection
│ ├── evaluation.py # Error analysis and feature importance
│ └── prediction.py # Demo predictions
│
├── results/
│ ├── figures/ # Saved plots and visualizations
│ └── models/ # Model outputs and evaluation tables
│
└── data/
└── players\_21.csv # Raw FIFA dataset

