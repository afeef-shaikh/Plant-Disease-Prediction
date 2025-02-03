# 🌱 Plant Disease Prediction - AI-Powered Image Classifier

![GitHub Repo Stars](https://img.shields.io/github/stars/afeef-shaikh/Plant-Disease-Prediction?style=social)
![GitHub Forks](https://img.shields.io/github/forks/afeef-shaikh/Plant-Disease-Prediction?style=social)
![GitHub Issues](https://img.shields.io/github/issues/afeef-shaikh/Plant-Disease-Prediction)
![GitHub License](https://img.shields.io/github/license/afeef-shaikh/Plant-Disease-Prediction)

🌿 **AI-powered plant disease detection** built using **TensorFlow, Streamlit, and Deep Learning** to help farmers and plant enthusiasts detect diseases in plants by simply uploading an image of a leaf.

---

## 🚀 Features
- 🌍 **AI-based Image Classification** - Identifies plant diseases using a trained deep learning model.
- 🎯 **High Accuracy** - Powered by a CNN model trained on a plant disease dataset.
- 📸 **Image Upload Support** - Users can upload an image of a plant leaf for instant diagnosis.
- 📊 **Prediction Confidence Scores** - Shows the probability of different disease classes.
- 🖥️ **Streamlit Web UI** - Simple and interactive UI for easy access.
- 💾 **Lightweight & Efficient** - Designed for quick and accurate predictions.
- 🐳 **Docker Support (Coming Soon!)** - Deploy using Docker for portability.

---

## 📷 Demo
![Plant Disease Prediction Demo](https://github.com/afeef-shaikh/Plant-Disease-Prediction/blob/main/static/demo.gif)  
🚀 **Try it out yourself!**

---

## 📦 Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/afeef-shaikh/Plant-Disease-Prediction.git
cd Plant-Disease-Prediction
```

### 2️⃣ Set Up a Virtual Environment
```sh
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### 3️⃣ Install Dependencies
```sh
pip install -r app/requirements.txt
```

### 4️⃣ Download the Model
Since GitHub does not allow large files, download the trained model manually:  
🔗 [**Google Drive Link**](https://drive.google.com/file/d/1eJr1kCEng9nliGRWdAK2fanyxw6V9iPP/view?usp=sharing)

After downloading, **move the file** to the correct location:
```sh
mv ~/Downloads/plant_disease_prediction_model.h5 app/trained_model/
```

### 5️⃣ Run the App
```sh
streamlit run app/main.py
```

🚀 Open **`http://localhost:8501`** in your browser and start predicting plant diseases!

---

## 🛠️ How It Works
1. **User Uploads an Image** 🌿  
   - Upload an image of a plant leaf.
2. **Model Preprocesses the Image** 🖼️  
   - Resizes and normalizes the image.
3. **Deep Learning Model Predicts Disease** 🤖  
   - Uses a trained CNN model.
4. **Results Are Displayed** 📊  
   - Shows the detected disease name and confidence scores.

---

## 📁 Project Structure
```
📂 Plant-Disease-Prediction
├── 📂 app
│   ├── 📂 static
│   ├── 📂 trained_model
│   │   ├── plant_disease_prediction_model.h5  # Trained Model (Download manually)
│   │   ├── class_indices.json  # Class mapping file
│   ├── main.py  # Streamlit app
│   ├── requirements.txt  # Dependencies
├── 📂 model_training_notebook
│   ├── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb  # Jupyter Notebook (Training)
├── 📂 test_images
│   ├── test_apple_black_rot.JPG  # Sample test image
├── .gitignore
├── README.md  # You are here!
```

---

## 🚀 Deployment Options
### 1️⃣ Local Deployment
Run the Streamlit app locally using:
```sh
streamlit run app/main.py
```

### 2️⃣ Docker Deployment (Coming Soon!)
We'll be adding **Docker support** for easy deployment across platforms.

---

## ⚡ Future Improvements
✅ Deploy on **Docker & Cloud Platforms**  
✅ Enhance **model accuracy with more training**  
✅ Add support for **more plant species**  
✅ Create an **API for mobile app integration**

---

## 🙌 Contributions
Contributions are **welcome**!  
If you find bugs or have feature requests, feel free to **open an issue** or submit a **pull request**.

1. **Fork the repo** 🍴  
2. **Create a branch** (`git checkout -b feature-name`) 🌱  
3. **Commit changes** (`git commit -m "Added new feature"`) 📝  
4. **Push** (`git push origin feature-name`) 🚀  
5. **Open a Pull Request** 🔥

---

## 📜 License
This project is licensed under the **MIT License**.

📌 **Disclaimer:** This model is trained on a limited dataset and should not be used for medical or commercial agricultural decisions.

---

## 📞 Contact
💬 Questions? Reach out!  
📧 Email: [achiever.afeef04@gmail.com](mailto:achiever.afeef04@gmail.com)  
🐙 GitHub: [@afeef-shaikh](https://github.com/afeef-shaikh)

---
