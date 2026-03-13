# Beyond Stated Preferences: Building a Better Match Predictor for Dating Apps

A behavioral analysis and match prediction model built on the Columbia Speed Dating dataset. 
The goal is not just to predict matches — it is to demonstrate that stated preferences 
are unreliable inputs, and that reciprocity signals are the real driver of mutual attraction.

---

## Business Context

A dating app is experiencing a decline in match rates. The current recommendation engine 
relies on stated preferences — what users declare they want during onboarding. 
This project tests whether those preferences actually predict real choices, and builds 
a better predictor from behavioral signals observed during encounters.

Three business questions driving the analysis:

- Do stated preferences predict actual choices?
- Do users who overestimate themselves match less?
- Can we predict a mutual match from behavioral signals?

---

## Key Findings

| Metric | Value |
|---|---|
| Model | XGBoost + decision threshold optimization |
| AUC-ROC | 0.881 |
| Recall | 0.819 |
| F1-score | 0.580 |
| Decision threshold | 0.40 |
| Match rate in dataset | 16.5% |
| Class imbalance ratio | 5.1:1 |

**Stated preferences are statistically indistinguishable from random:**
Spearman rank correlation between stated and revealed preference rankings: ρ = 0.37, p = 0.47.
Intelligence and sincerity are overweighted. Fun and shared interests are underweighted.

**71% of participants overestimate their attractiveness:**
Mean self-rating: 7.06 / Mean rating received: 6.16 — a gap of 0.90 points.
This creates misaligned expectations and drives early churn.

**Reciprocity dominates all other signals:**
The top 4 predictive features are all engineered reciprocity scores.
`recip_overall` alone accounts for a 0.057 drop in AUC-ROC when removed —
more than all other features combined.

---

## Model Performance

| Model | AUC-ROC | F1 | Recall | Precision |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.851 | 0.548 | 0.812 | 0.413 |
| XGBoost (threshold=0.5) | 0.881 | 0.576 | 0.736 | 0.473 |
| **XGBoost (threshold=0.4)** | **0.881** | **0.580** | **0.819** | **0.449** |

Selected model: XGBoost with threshold=0.40.
Rationale: highest AUC-ROC + recall aligned with business priority.
In a dating app context, missing a real match (false negative) is more damaging 
than a bad recommendation (false positive).

---

## Project Structure
```
speed-dating-match-prediction/
├── data/                          # Raw dataset (not versioned)
│   └── Speed_Dating_Data.csv
├── notebooks/
│   └── speed_dating_analysis.ipynb
├── outputs/                       # Charts and model outputs
│   ├── missing_values_distribution.png
│   ├── target_analysis.png
│   ├── stated_vs_revealed.png
│   ├── overestimation_bias.png
│   ├── temporal_preference_shift.png
│   ├── roc_comparison.png
│   ├── threshold_optimization.png
│   ├── feature_importance.png
│   └── permutation_importance.png
├── presentations/                 # Business presentation (coming soon)
├── src/                           # Reusable modules (future)
└── README.md
```

---

## Methodology

- Section 0 — Setup and data loading
- Section 1 — Business framing, error cost matrix, stakeholder mapping
- Section 2 — EDA: target analysis, stated vs. revealed preferences, overestimation bias, temporal dimension
- Section 3 — Feature engineering: reciprocity scores, overestimation score, age difference
- Section 4 — Modeling: Logistic Regression baseline, XGBoost, threshold optimization
- Section 5 — Feature importance: XGBoost native importance, permutation importance
- Section 6 — Business recommendations: 3 actionable product levers

---

## Stack

Python 3.10
pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

---

## Dataset

Columbia Business School Speed Dating Experiment (Fisman & Iyengar, 2006)
551 participants — 8,378 encounters — 21 experimental waves
Available on [Kaggle](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment)

---

## Reproduce
```bash
git clone https://github.com/ArvindB75/speed-dating-match-prediction
cd speed-dating-match-prediction
conda create -n speed-dating python=3.10
conda activate speed-dating
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
# Add Speed_Dating_Data.csv to data/ directory
jupyter notebook notebooks/speed_dating_analysis.ipynb
```
