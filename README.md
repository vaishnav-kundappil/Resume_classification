# Resume Classification using Machine Learning

## 📌 Project Overview
This project focuses on automating the classification of resumes using **Natural Language Processing (NLP) and Machine Learning**. The application allows users to upload resumes in **PDF, DOCX, or DOC** formats and predicts their category based on the content. The final model is deployed using **Streamlit**.

## 🔍 Workflow
1. **Exploratory Data Analysis (EDA)**
   - Performed data exploration and visualization.
   - Analyzed word distributions and keyword significance.
2. **Data Preprocessing**
   - Text cleaning: Removed stopwords, punctuation, and special characters.
   - Tokenization and vectorization using **TF-IDF**.
3. **Model Building**
   - Evaluated multiple machine learning models:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)
     - KNN
     - Decision Tree
     - Bagging
     - AdaBoost
     - Naïve Bayes (Final Model)
   - Hyperparameter tuning to optimize performance.
4. **Deployment with Streamlit**
   - Developed a web-based interface to upload resumes and get classification results.

## 🚀 Features
- Supports **PDF, DOCX, and DOC** file uploads.
- Uses **NLP techniques** to preprocess resume content.
- Classifies resumes into different categories using a **Naïve Bayes model**.
- Provides an interactive **Streamlit** UI for user-friendly experience.

## 🏗️ Project Structure
```
Resume_Classification_Project/
│-- data/
│   ├── resumes.csv  # Dataset used for training
│-- notebooks/
│   ├── EDA_and_Model_Building.ipynb  # Jupyter Notebook with all steps
│-- models/
│   ├── resume_classifier.pkl  # Trained Naïve Bayes model
│   ├── vectorizer.pkl  # TF-IDF vectorizer
│-- app/
│   ├── resume_classification_app.py  # Streamlit web app
│-- README.md
│-- requirements.txt  # Required libraries
```

## 🔧 Installation & Setup
1. Clone the repository:
   ```sh
   git clone [GitHub_Repo_Link]
   cd Resume_Classification_Project
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run app/resume_classification_app.py
   ```

## 📜 Example Usage
1. Upload a **resume file** (PDF, DOCX, or DOC).
2. The model extracts text and preprocesses it.
3. The classified category is displayed instantly.

## 📌 Technologies Used
- **Python** (pandas, numpy, re, nltk, scikit-learn)
- **Machine Learning** (Naïve Bayes, TF-IDF)
- **NLP** (Text preprocessing, tokenization, stopword removal)
- **Jupyter Notebook** (EDA, Model Training)
- **Streamlit** (Deployment)

## 📬 Contact
For any issues or suggestions, feel free to reach out!

📧 vaishnavkundappil@gmail.com

