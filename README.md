# 💧 AI-Powered Water Potability Predictor

## Overview
This project predicts whether water is safe to drink (potable) based on its chemical properties, using a machine learning model. The goal is to provide a low-cost, accessible digital tool to help communities and individuals assess water safety, supporting UN SDG 6: Clean Water and Sanitation.

## Motivation
Access to clean water is a fundamental human right, yet millions lack reliable means to test water quality. By leveraging open data and AI, this project aims to democratize water safety analysis and raise awareness about water quality issues.

## 🧪 Dataset
- **Source:** [Kaggle – Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- **Description:** The dataset contains water samples with various chemical properties and a binary label indicating potability (1 = safe, 0 = not safe).
- **Features Used:**
  - `pH`: Acidity or alkalinity of water
  - `Hardness`: Calcium and magnesium content
  - `Solids`: Total dissolved solids (ppm)
  - `Chloramines`: Amount of chloramines (ppm)
  - `Sulfate`: Sulfate concentration (mg/L)
  - `Conductivity`: Electrical conductivity (μS/cm)
  - `Organic Carbon`: Organic carbon content (ppm)
  - `Trihalomethanes`: Trihalomethanes concentration (μg/L)
  - `Turbidity`: Water cloudiness (NTU)

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/water-quality-ai.git
   cd water-quality-ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Start the Streamlit app:
```bash
streamlit run app.py
```

## 🧠 Model & Methodology
- **Algorithm:** The model is built using Scikit-learn (e.g., Random Forest, Logistic Regression, or similar).
- **Training:** Trained on the Kaggle dataset after preprocessing (handling missing values, scaling, etc.).
- **Deployment:** The trained model is saved as `water_quality_model.pkl` and loaded by the Streamlit app for real-time predictions.

## 📈 Example Usage
1. Launch the app.
2. Enter the chemical properties of your water sample in the provided fields.
3. Click 'Predict'.
4. The app will display whether the water is likely potable or not.

**Example Input:**
- pH: 7.2
- Hardness: 180
- Solids: 12000
- Chloramines: 7.5
- Sulfate: 330
- Conductivity: 420
- Organic Carbon: 10
- Trihalomethanes: 80
- Turbidity: 3.5

**Example Output:**
> "Prediction: The water is likely safe to drink."

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 📬 Contact
For questions or suggestions, please open an issue or contact [marksabuto@gmail.com](mailto:marksabuto@gmail.com).

---
This project supports UN SDG 6: Clean Water and Sanitation by providing a low-cost digital tool to detect water safety.