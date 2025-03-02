# ✈️ Flight Price Prediction

This is a **Machine Learning-based Flight Price Prediction** web app built with **Streamlit**. The model predicts the price of a flight ticket based on various parameters such as airline, source, destination, duration, and more.

---

## 📌 Features
✅ Load and preprocess flight price dataset  
✅ Train multiple ML models (Linear Regression, Random Forest, XGBoost)  
✅ Automatically selects the best model based on R² score  
✅ Saves the best model using `joblib`  
✅ Streamlit-based UI for user-friendly price prediction  
✅ **Prevents auto-reloading on filter change**  
✅ Uses `st.session_state` to update only on **button click**  

---

## 🧀 Installation

### **1⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/Flight-Price-Prediction.git
cd Flight-Price-Prediction
```

### **2⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### **3⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### **1⃣ Train the Model**
Run the following command to train the model and save the best one:
```bash
python app.py
```
This will:
- Load and preprocess the dataset
- Train multiple models (`LinearRegression`, `RandomForest`, `XGBoost`)
- Select the best-performing model based on R² score
- Save the model (`best_model.pkl`) and encoders (`label_encoders.pkl`)

### **2⃣ Run the Web App**
```bash
streamlit run app.py
```

---

## 🎯 How It Works
1. Select the **Airline, Source, Destination, Route, Additional Info**  
2. Adjust **Departure Time, Arrival Time, Duration, and Stops**  
3. Click the **"Predict Price"** button  
4. The model predicts the flight price and displays it  

💡 **The app will not reload when you change filters. It only predicts when you click the button.**  

---

## 📂 Project Structure
```
📚 Flight-Price-Prediction
│── 📄 app.py                  # Streamlit app
│── 📄 requirements.txt         # Required dependencies
│── 📄 Flight_Price.csv         # Dataset
│── 📄 best_model.pkl           # Trained model
│── 📄 label_encoders.pkl       # Encoders for categorical features
│── 📄 README.md                # Project documentation
```

---

## 📌 Dependencies
- **Python 3.8+**
- `pandas`
- `numpy`
- `streamlit`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `joblib`
- `mlflow`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## 📌 Acknowledgments
- Dataset: [Flight Price Dataset](https://drive.google.com/file/d/1RrSe3M0Ia-ekihZzWZXLTYFa-_bsba-C/view?usp=sharing)
- ML Framework: **scikit-learn, XGBoost**
- UI: **Streamlit**

---

## 📌 License
This project is **open-source** under the MIT License.

---

