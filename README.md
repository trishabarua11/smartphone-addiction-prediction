 # Smartphone Addiction Prediction
A machine learning project that predicts smartphone addiction levels (**Low, Medium, High**) based on user phone usage behavior.  
The model is trained using a **Random Forest Classifier** and deployed as an interactive **Streamlit web application**.

# Features
- Predicts smartphone addiction level in real time
- Uses behavioral usage data
- Clean and simple Streamlit UI
- Trained and tested machine learning model

# Model
- Algorithm: Random Forest Classifier
- Type: Multi-class classification
- Classes:Low, Medium, High

# Input Parameters
The prediction is based on the following inputs:
- Daily phone usage (hours)
- Sleep hours per day
- Phone checks per day
- Number of apps used daily
- Time spent on social media (hours/day)
- Time spent on gaming (hours/day)

## Workflow
1. Dataset loading and preprocessing  
2. Feature encoding and scaling using StandardScaler 
3. Train-test split  
4. Model training using Random Forest  
5. Model evaluation (Accuracy, Precision, Recall, F1-score, Confusion Matrix)  
6. Model saving (.pkl files)  
7. Deployment using Streamlit  

