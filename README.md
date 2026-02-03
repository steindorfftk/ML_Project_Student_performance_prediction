# From Linear Models to Advanced Ensembles: A Kaggle Regression Journey

This repository documents my solution and learning process from my **first Kaggle regression competition**.  
The goal of the competition was to **predict students’ exam scores** using a large synthetic dataset.

More than achieving a competitive leaderboard position, this project focuses on **modeling decisions, feature engineering, ensembling strategies, and validation at scale**.

---

## 🧠 Problem Overview

- **Task**: Regression — predict exam scores for students  
- **Dataset**: Synthetic
  - ~630,000 training samples
  - ~270,000 test samples
  - 10 features
- **Evaluation metric**: RMSE

---

## 🔍 Initial Approach

I started with a standard workflow:
- Exploratory Data Analysis (EDA)
- Feature selection based on domain intuition
- Benchmarking **8 different regression algorithms**

Among the initial models, **Elastic Net** achieved the lowest RMSE.  
I then applied **Optuna** for hyperparameter optimization, but the result was suboptimal:

- **RMSE**: 8.94  
- **Leaderboard position**: ~1691

Keeping all features and applying **ordinal encoding** resulted in only marginal improvements (LB ≈ 1672).

---

## 🚀 Feature Engineering Breakthrough

The first major performance gain came from **feature engineering**, which significantly changed the structure of the data.

Two engineered features proved especially effective:

- **attendance_study_ratio** = `study_hours × class_attendance`
- **sleep_study_balance** = `study_hours × sleep_hours`

Both showed an **almost linear relationship with the target**, and more importantly, shifted model performance in favor of **tree-based boosting methods** over linear models.

---

## 🌲 Boosting Models & Cross-Validation

With the new features:
- A first **XGBoost** model achieved:
  - **RMSE** = 8.72
  - **LB** ≈ 845

- A **10-fold CV ensemble of XGBoost** further improved results:
  - **RMSE** = 8.71
  - **LB** ≈ 706

---

## 🤝 Advanced Ensembling

Next, I built stronger ensembles:
- **XGBoost + LightGBM + CatBoost**
- 10-fold cross-validation per model
- Ensemble weights optimized using a **hill-climbing strategy**

This reduced the error to:
- **RMSE** = 8.70899  
- **Leaderboard position** ≈ 606

---

## 🔧 Additional Feature Transformations

Further feature engineering led to another significant improvement.  
Two transformations of `study_hours` were particularly impactful:

- `sin(study_hours)`
- `log(study_hours)`

With these additions:
- **RMSE** = 8.67446  
- **Leaderboard position** ≈ 356

---

## 📊 Target Distribution Adjustment

An interesting observation during EDA was the presence of **hard truncation at the extremes of the target distribution** (sharp cutoffs in the histogram tails).

Applying **prediction clipping**:
- Max prediction = 100
- Min prediction ≈ 19.6

This slightly improved the RMSE, although it did not translate into a leaderboard gain.

---

## 🧪 Pseudo-Labeling with Deep Learning

The final and most impactful contribution came from **pseudo-labeling**:
1. A deep learning model was trained on the original training set
2. Targets for the test set were estimated
3. The final ensemble (**XGBoost + LightGBM + CatBoost**) was retrained using:
   - **50-fold CV per model**
   - Augmented data with pseudo-labels

### ✅ Final Results
- **RMSE**: 8.66797  
- **Leaderboard**: Top 20% of the competition

---

## 📈 Key Takeaways

- Feature engineering can dramatically reshape model performance
- Boosting models thrive when interactions are properly exposed
- Cross-validation and ensembling provide consistent, reliable gains
- Pseudo-labeling can be highly effective even in regression problems
- Large-scale validation requires careful design and computational discipline

---

## 🙌 Final Thoughts

This competition was an excellent hands-on introduction to **regression modeling at scale**.  
Beyond the final ranking, the real value of this project lies in the practical insights gained around:
- Feature engineering
- Model selection
- Ensembling strategies
- Robust validation

Looking forward to applying these lessons to future competitions and real-world problems.
