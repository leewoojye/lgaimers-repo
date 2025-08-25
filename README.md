# LG AI Hackathon: AI Gamers

This repository contains code and resources for the [2024 LG AI Hackathon: AI Gamers](https://dacon.io/competitions/official/236559/overview/description) competition hosted on DACON. The goal of this competition is to develop AI models that can predict the next move in a turn-based game, using provided game logs and player information.

---

## Competition Overview

- **Host:** LG AI Research
- **Objective:** Predict the next move in a turn-based game based on historical game logs and player data.
- **Data:** Includes game logs, player statistics, and other relevant features.
- **Evaluation Metric:** Accuracy of the predicted moves.

For more details, please refer to the [official competition page](https://dacon.io/competitions/official/236559/overview/description).

---

## Project Structure

```
lgaimers-repo/
│
├── data/                # Raw and processed datasets
│   ├── raw/
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
    - Place raw data files in the `data/raw/` directory.

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