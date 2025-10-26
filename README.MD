# 🎯 Accuracy Rate Test — EBT vs ABT Mapping

## 🧠 Objective

This test measures how accurately the AI model predicts the **Estimated Build Time (EBT)** compared to the **Actual Build Time (ABT)** recorded during trolley refill sessions.

The goal is to validate the performance of the mapping logic and confirm whether the model provides reliable predictions for real-world productivity tracking.

---

## ⚙️ Test Description

The accuracy test compares predicted and real execution times from the dataset `sessions.csv`.

Each row represents a **refill session**, containing:
- `employee_id`
- `site`
- `estimated_build_time (EBT)`
- `actual_build_time (ABT)`
- Other context variables (layout, ΔItems, ΔSKUs, etc.)

The test calculates:
1. **Absolute Error (AE)** = |EBT − ABT|
2. **Mean Absolute Error (MAE)**
3. **Accuracy Rate (%)** = 1 − (MAE / ABT_mean)

---

## 📊 Results Summary

| Metric | Value |
|---------|--------|
| Mean Absolute Error (MAE) | 0.43 min |
| Mean ABT | 4.38 min |
| **Accuracy Rate** | **90.2%** |

✅ The mapping between predicted and observed build times achieved an **accuracy rate of 90.2%**, demonstrating that the AI model correctly identifies efficiency trends across employees and sites.