# LG AI Hackathon: Resort Menu Prediction

This repository contains code and resources for the [2025 LG AI Hackathon: Resort Menu Prediction](https://dacon.io/competitions/official/236559/overview/description) competition hosted on DACON.  
The goal of this competition is to develop AI models that can accurately predict the menu choices of resort guests based on their reservation and demographic information.

---

## Competition Overview

- **Host:** LG AI Research
- **Objective:** Predict the menu selection for each guest at a resort using reservation data, guest demographics, and historical menu choices.
- **Data:** Includes guest reservation records, demographic features, and menu selection logs.
- **Evaluation Metric:** Accuracy of menu prediction for each guest.

For more details, please refer to the [official competition page](https://dacon.io/competitions/official/236559/overview/description).

---

## Project Structure

```
lgaimers-repo/
│
├── data/                # Raw and processed datasets
│   ├── test/
│   ├── train/
│   └── processed/
│
├── eda/                 # Exploratory Data Analysis notebooks
│   └── eda_woojye.ipynb
│
├── models/              # Model training and inference scripts
│   ├── random_forest.py
│   ├── random_forest_requirements.txt
│   ├── xgboost.py
│   ├── xgboost_requirements.txt
│   ├── lgbm.py
│   ├── lgbm_requirements.txt
│   ├── lstm.py
│   ├── lstm_requirements.txt
│   ├── prophet_test.ipynb
│   └── prophet_requirements.txt
│
├── utils/               # Utility functions (preprocessing, visualization, etc.)
│   └── preprocessing.py
│
├── results/             # Model outputs, reports, and plots
│
├── README.md
└── requirements.txt     # (Empty) Global requirements file
```

---

## Model-specific Requirements

Each model has its own requirements file under the `models/` directory.  
To install dependencies for a specific model, run:

```bash
pip install -r models/<model_name>_requirements.txt
```

For example, to install requirements for XGBoost:
```bash
pip install -r models/xgboost_requirements.txt
```

---

## Getting Started

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/lgaimers-repo.git
    cd lgaimers-repo
    ```

2. **Download the competition data**
    - Place raw data files in the `data/train/` and `data/test/` directories.

3. **Set up your environment**
    - Install the requirements for the model you want to use (see above).

4. **Run EDA**
    - Explore the data using notebooks in the `eda/` directory.

5. **Train and Evaluate Models**
    - Use scripts in the `models/` directory to train and evaluate different models.

---

## Notes

- The global `requirements.txt` is intentionally left empty.  
  Please use the model-specific requirements files for dependency management.
- Utility functions for preprocessing and visualization are located in the `utils/` directory.
- Results, including model outputs and reports, are saved in the `results/` directory.

---

## Contact

For questions or collaboration, please open an issue or contact the repository owner.

---