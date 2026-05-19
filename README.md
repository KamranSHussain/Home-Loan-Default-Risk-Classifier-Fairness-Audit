```markdown
# Sex-Based Algorithmic Fairness & Causal AI Risk Audit of Credit Risk ADS

This repository contains a comprehensive algorithmic fairness audit and causal risk analysis of an Automated Decision System (ADS) developed for the **Home Credit Default Risk** Kaggle competition. Since access to credit is a vital pillar of economic mobility, this project evaluates individual and group fairness across sex demographics to identify bias, quantify uncertainty, and map causal mechanisms within complex machine learning frameworks.

## Repository Structure

* **`Credit Risk ADS Audit.pdf`**: The formal technical report detailing the motivation, comprehensive background, end-to-end engineering pipeline, algorithmic evaluation, fairness findings, and recommended post-processing mitigations.
* **`fairness_audit.ipynb`**: The fully executable Jupyter notebook implementing the entire pipeline—from high-dimensional feature engineering and model optimization to advanced fairness audits using `Fairlearn`, causal inference using `DoWhy`, and confidence estimation via `Split Conformal Prediction`.

---

## Methodology & Technical Core

### 1. Algorithmic Fairness Audit (`Fairlearn`)
* Evaluated group fairness metrics across sex demographics to locate historical bias leakage.
* Tracked **5 core fairness indicators**, explicitly uncovering critical disparities in group error rates—such as **Equalized Odds** violations, False Positive Rate (FPR) deviations, and **False Negative Rate (FNR) differences**.

### 2. Causal AI & Counterfactual Fairness (`DoWhy`)
* Leveraged **`DoWhy`** to estimate the **Average Treatment Effect (ATE)** of sex on automated credit risk predictions using the `identify_effect` framework.
* Architected a **Structural Causal Model (SCM)** to rigorously evaluate **counterfactual fairness**, isolating the exact causal pathways of protected attributes on credit classifications to ensure decision-making remains grounded in objective financial behavior.

### 3. Uncertainty Quantification
* Deployed a **Split Conformal Prediction** framework to quantify and map **epistemic uncertainty** across gender groups.
* Derived exact, distribution-free prediction intervals that demonstrated hidden variances in model confidence (e.g., higher average prediction set sizes for male applicants), ensuring statistical reliability before deployment.

---

## 🚀 Installation & Prerequisites

To set up your local environment and reproduce the fairness audit, ensure you have Python 3.10+ installed along with the required libraries.

```bash
# Clone the repository
git clone [https://github.com/your-username/credit-risk-fairness-audit.git](https://github.com/your-username/credit-risk-fairness-audit.git)
cd credit-risk-fairness-audit

# Install required packages
pip install pandas numpy scikit-learn lightgbm fairlearn shap dowhy matplotlib seaborn
