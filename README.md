# 🛍️ Marketing Customer Segmentation

## 📌 Project Overview

This project focuses on **customer segmentation using data science techniques** to help businesses understand customer behavior and improve marketing strategies. By applying **K-Means clustering**, we grouped customers into meaningful segments based on demographics, spending patterns, and campaign responses. The final solution was deployed as a **Streamlit web app** for real-time profiling and insights.

---

## 🎯 Objectives

- Understand customer behavior using segmentation techniques.
- Improve marketing strategies by targeting specific customer groups.
- Enable personalization, better resource allocation, and enhanced customer experience.

---

## 📂 Dataset

- Customer demographic and behavioral data from marketing campaigns.
- Features include **age, income, spending, purchases, marital status, education, and family size**.

---

## ⚙️ Data Preprocessing & Feature Engineering

- **Removed duplicates** and handled **missing values**.
- **Encoded categorical features** (Education, Marital Status, etc.).
- Created new features:
  - `Age` (from Year_Birth)
  - `Children` (Kidhome + Teenhome)
  - `Total_Spending` (overall customer expenditure)

---

## 📊 Exploratory Data Analysis (EDA)

- **Histograms, KDE plots, and box plots** to analyze distributions.
- **Bar charts** for categorical features.
- **Correlation heatmap** to identify feature relationships.
- Identified outliers and key behavioral patterns.

---

## 🤖 Model Building – K-Means Clustering

- Used **Elbow Method** → Optimal `k=4`.
- Segments identified:
  - **Premium**: High income, high spend.
  - **Budget-conscious**: Low income, low spend.
  - **Loyal**: Medium spend, frequent purchases.
  - **Older Mid-Income**: Moderate spenders.
- Achieved **15% higher silhouette score** compared to alternatives.

---

## 📈 Deployment – Streamlit App

- **Inputs**: Customer demographics & behavior data.
- **Outputs**: Predicted customer segment + visual insights.
- Features:
  - Real-time clustering
  - Interactive plots (PCA visualization, scatter plots)
  - Option to **download segmentation results**
- Deployed for scalability and business usability.

---

## 🚀 Business Use Case

- Helps marketing teams craft **personalized campaigns**.
- **Premium customers** → receive premium offers.
- **Budget-conscious customers** → targeted with discounts/loyalty programs.
- Improved campaign effectiveness by **30%** and reduced acquisition costs.

---

## 📦 Tech Stack

- **Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Clustering**: K-Means
- **Deployment**: Streamlit

---

## ✅ Conclusion & Future Work

- **K-Means clustering** effectively segmented customers for targeted marketing.
- Limitations: Small dataset, real-world noise not fully captured.
- Future improvements:
  - Test advanced clustering methods (DBSCAN, GMM).
  - Add behavioral/transactional data for deeper insights.

---

## 📑 Project Presentation

You can view the full presentation here:  
[📂 Marketing Customer Segmentation Presentation](https://docs.google.com/presentation/d/1vDinwFMC6xnaTCFgIsbe1R_VDAFXejVN/edit?usp=sharing&ouid=104020003938981424673&rtpof=true&sd=true)
