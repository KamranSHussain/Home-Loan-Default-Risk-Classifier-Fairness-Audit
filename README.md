# Sex-Based Algorithmic Fairness & Causal AI Mitigation of Credit Risk ADS

This repository contains an algorithmic fairness audit, optimization pipeline, and causal risk mitigation of an Automated Decision System (ADS) developed for the **Home Credit Default Risk** Kaggle competition. 

While access to credit is a vital pillar of economic mobility, baseline historical training data frequently embeds systemic biases. This project addresses both **disparate treatment** and **disparate impact** across sex demographics—auditing group fairness, resolving proxy attribute leakage, restructuring classification thresholds, and causally validating the final model to deliver a legally compliant, production-ready system.

---

## Repository Structure

* **`Credit Risk ADS Audit.pdf`**: The formal technical report detailing the business motivation, baseline disparities, mathematical mitigation pipeline, and regulatory compliance validation.
* **`fairness_audit.ipynb`**: The fully executable Jupyter notebook implementing the entire engineering workflow—including high-dimensional feature processing, Fairlearn in-processing, threshold post-processing, DoWhy causal graphs, and Split Conformal Prediction.

---

## Methodology & Technical Core

### 1. Algorithmic Mitigation & Group Fairness (`Fairlearn`)
The baseline LightGBM model exhibited severe demographic disparities due to historical proxy leakage and data undersampling. To resolve both **disparate treatment** and **disparate impact** without dismantling the model's core predictive utility, a comprehensive three stage intervention pipeline was deployed:

* **Pre-Processing (Fairness Through Blindness)**: Stripped explicitly protected attributes—including sex, age, and marital status—directly from the training feature set to ensure baseline legal compliance. However, because high-dimensional credit datasets contain dense proxy variables (e.g., income structures, job categories) that allow complex trees to implicitly reconstruct demographic identity, blindness alone was insufficient and required algorithmic intervention.
* **In-Processing Mitigation**: Implemented Fairlearn's **Exponentiated Gradient** reduction algorithm constrained for **Equalized Odds** ($\epsilon = 0.02$). This mathematically penalized the model for relying on hidden proxy structures, successfully collapsing the demographic False Negative Rate (FNR) difference from **12.1 to 2.1 percentage points**, and the False Positive Rate (FPR) difference from **13.67 to 3.93 percentage points**.
* **Post-Processing Optimization**: Navigated the fairness-accuracy trade-off by dynamically shifting the final decision boundary to a **0.45 threshold**. This step recovered lost default recall back to **70%** to protect institutional capital, while boosting the overall **Equalized Odds Ratio from 0.62 to a regulatory-compliant 0.87**.
* **Demographic Parity**: Lifted the Demographic Parity Ratio from **0.63 to 0.85**, compressing the overall selection rate difference down to **5.05 percentage points** to ensure both error rates and selection rates were equitably balanced.

### 2. Causal AI & Counterfactual Fairness (`DoWhy`)
Observational group metrics alone cannot prove a model has stopped using hidden proxy variables. To establish mathematical proof of fairness, we mapped the system into a **Structural Causal Model (SCM)** using Directed Acyclic Graphs (DAGs).
* **ATE Minimization**: Isolated and estimated the Average Treatment Effect (ATE) of an applicant's sex on final classifications. The interventions successfully neutralized the baseline causal effect of **-0.079** down to a statistically negligible **-0.008**. 
* **Causal Refutation Testing**: Subjected the SCM to a comprehensive trio of invariance stress tests to confirm stability. Introducing a **Random Common Cause** left the true effect virtually unchanged at -0.0079 (p-value = 0.94); bootstrapping a 90% **Data Subset** proved data-distribution stability with an effect of -0.0075 (p-value = 0.66); and substituting a **Placebo Treatment** collapsed the effect to absolute zero (0.000066, p-value = 0.88). These combined results fail to reject their respective null hypotheses, statistically validating that the estimated ATE is robust.

### 3. Uncertainty Quantification (`Split Conformal Prediction`)
To audit how confidently the model treats different demographic groups, we deployed a **Split Conformal Prediction** framework.
* **Epistemic Uncertainty Mapping**: Derived exact, distribution-free prediction intervals independent of model architecture. 
* **Confidence Inequality Insights**: Uncovered that the baseline model carried higher epistemic uncertainty when evaluating male applicants (average prediction set size of 1.74 vs 1.54 for females), providing crucial context regarding data density and group-level variance prior to deployment.

---

## Performance Summary

| Metric | Baseline Model | Mitigated Model (Production) | Business / Ethical Impact |
| :--- | :--- | :--- | :--- |
| **Equalized Odds Ratio** | 0.62 | **0.87** | Exceeds standard regulatory compliance thresholds |
| **FNR Disparity Gap** | 12.12% | **2.13%** | Equitably distributes missed-default risk |
| **FPR Disparity Gap** | 13.67% | **3.93%** | Eliminates disproportionate false accusations |
| **Causal ATE (Sex)** | -0.0797 | **-0.0080** | Eliminates direct/indirect causal impact of gender |
| **Default Recall** | 69.08% | **69.55%** | Retains full capability to flag high-risk defaults |
| **Overall Accuracy** | 72.63% | **72.42%** | Achieved fairness with a negligible 0.21% utility tax |

---

## Installation & Prerequisites

To set up your local environment and reproduce the audit and mitigation pipeline, ensure you have Python 3.10+ installed along with the required libraries.

```bash
# Clone the repository
git clone [https://github.com/your-username/credit-risk-fairness-audit.git](https://github.com/your-username/credit-risk-fairness-audit.git)
cd credit-risk-fairness-audit

# Install required packages
pip install pandas numpy scikit-learn lightgbm fairlearn shap dowhy matplotlib seaborn networkx
