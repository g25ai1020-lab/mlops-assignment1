# ML Ops Assignment 1: Boston Housing Price Prediction

## Project Description
This project implements a complete machine learning workflow to predict house prices using the Boston Housing dataset with multiple classical ML models (Decision Tree Regressor and Kernel Ridge Regression).

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Conda (recommended)

### Steps to Run

#### 1. Clone the repository
```bash
git clone https://github.com/g25ai1020-lab/mlops-assignment1.git
cd mlops-assignment1
```

#### 2. Create a conda environment
```bash
conda create -n mlops python=3.8 -y
conda activate mlops
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run Decision Tree Model (from dtree branch)
```bash
git checkout dtree
python train.py
```

#### 5. Run Kernel Ridge Model (from kernelridge branch)
```bash
git checkout kernelridge
python train2.py
```

#### 6. Run both models together
On kernelridge branch, both models will run via:
```bash
python train.py
python train2.py
```

## Project Structure
```
mlops-assignment1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── misc.py                            # Shared utility functions
├── train.py                           # Decision Tree Regressor
├── train2.py                          # Kernel Ridge Regressor
└── .github/workflows/
    └── ml-pipeline.yml                # GitHub Actions CI/CD
```

## Models Implemented

### 1. Decision Tree Regressor
- **Branch:** `dtree`
- **File:** `train.py`
- **Description:** Classic tree-based regression model from scikit-learn
- **Performance:** MSE on test set displayed after execution

### 2. Kernel Ridge Regression
- **Branch:** `kernelridge`
- **File:** `train2.py`
- **Description:** Regularized least squares with RBF kernel
- **Performance:** MSE on test set displayed after execution

## Dataset
**Boston Housing Dataset**
- **Samples:** 506
- **Features:** 13 (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)
- **Target:** MEDV (Median home values)
- **Source:** http://lib.stat.cmu.edu/datasets/boston

## Branch Information

| Branch | Purpose | Files |
|--------|---------|-------|
| `main` | Main development branch | README.md |
| `dtree` | Decision Tree implementation | requirements.txt, misc.py, train.py |
| `kernelridge` | Kernel Ridge + CI/CD | train2.py, .github/workflows/ml-pipeline.yml |

## GitHub Actions Workflow
- **Trigger:** Automatic push to `kernelridge` branch
- **Steps:**
  1. Check out code
  2. Set up Python 3.8
  3. Install dependencies
  4. Run Decision Tree model
  5. Run Kernel Ridge model
  6. Display summary

**Access logs:** GitHub → Repository → Actions → ML Pipeline

## Results & Performance
See GitHub Actions logs for training and testing results of both models.

MSE (Mean Squared Error) is displayed for each model on the test set.

## Author
Roll No: g25ai1020-lab

## Notes
- All branches must be preserved at submission
- Repository is public as per assignment requirements
- Git operations performed via command line only (no web upload)