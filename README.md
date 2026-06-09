# 📘 Health & Lifestyle Classification

## 📌 Descrizione
Progetto di classificazione binaria su un dataset sintetico di **100.000 campioni** e **48 features** relative a salute, abitudini e stile di vita.  
Obiettivo: prevedere lo stato di salute (*healthy* vs *diseased*).

https://www.kaggle.com/datasets/mahdimashayekhi/disease-risk-from-daily-habits
---

## 📂 Dataset
- 48 feature (numeriche, categoriche, binarie, ordinali)  
- Target sbilanciato (~70% healthy / 30% diseased)  
- Criticità principali: feature costanti, ridondanze, valori negativi, missing eterogenei, scale molto diverse.

---

## 🧹 Preprocessing
- Rimozione feature non informative  
- Imputazione: mediana (numeriche), categoria esplicita (categoriche)  
- Correzione valori negativi  
- Encoding: binario, ordinal, One‑Hot (`drop_first=True`)  
- Normalizzazione con MinMaxScaler  

---

## 🧪 Feature Engineering
Feature derivate per catturare interazioni non lineari:  
`smoking_stress_level`, `glucose_to_sugar_ratio`, `hydration_per_weight`, `sedentary_index`, ecc.

---

# 🔧 Pipeline utilizzate

## **1️⃣ Logistic Regression**
Pipeline utilizzata: [ MinMaxScaler → LogisticRegression(class_weight='balanced') ]

- GridSearchCV per iperparametri   
- Risultati:
  - AUC ROC ≈ 0.50  
  - AUPRC ≈ 0.30
    
→ Underfitting: nessun pattern utile appreso.



---

## **2️⃣ MLPClassifier**
Pipeline utilizzata: [ MinMaxScaler → SMOTE → MLPClassifier ]

- Best model: 2 hidden layers (64,64), ReLU, α=0.001
- - Risultati:
  - Training AUC ≈ 0.65  
  - Validation AUC ≈ 0.50 
→ Il modello non generalizza.

---

## 📊 Analisi non supervisionata
- **PCA**: nessuna separazione tra classi  
- **K-Means**: cluster con distribuzione target invariata  
→ Il dataset non mostra struttura predittiva.

---

## 🧾 Conclusioni
- Le feature disponibili **non contengono informazione sufficiente** per predire il target.  
- Modelli lineari e non lineari falliscono.  
- PCA e clustering confermano l’assenza di separabilità.
