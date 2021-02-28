# Risk Models for Medical Prognosis

## Outline

## What is medical prognosis?
Prognosis is a medical term that refers to predicting the risk of a future event. Here, event is a general term that captures a variety of things that can happen to an individual. Events can include outcomes such as death and other adverse events like a heart attack or a stroke, which might be risks for patients who have a specific medical condition or for the general population. 

Making prognosis is a clinically useful task for a variety of reasons:
 - First, prognosis is useful for informing patients their risk of developing an illness. For example, there are blood tests that are used to estimate the risk of developing breast and ovarian cancer. 
 - Second, prognosis are also useful for informing patients how long they can expect to survive with a certain illness. An example of this is cancer staging, which gives an estimate of the survival time for patients with that particular cancer. 

Prognosis is also useful for guiding treatment. In clinical practice, the prediction of the 10-year risk of heart attack is used to determine whether a patient should get drugs to reduce the risk. Another example is the six-month mortality risk. This is used for patients with terminal conditions that have become advanced and uncurable and is used to determine who should receive end-of-life care. 

This case study is about building such prognostic models to ***predict the 10-year risk of death of an individuals from the NHANES-I epidemiology dataset*** (for a detailed description of this dataset is given at the [CDC Website](https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx/)).

## Data Description
Set of features --> patient profile (includes clinical history, physical examinations and labs and imaging)
Target (Risk score) --> Computed from the features (risk equation = linear combination of natural log of features and its coefficients (coefficients is the factor of contribution of the feature)). Here natural log is taken to make equation linear.

 1. **Age:** Age of the patient
 2. **Diastolic BP:** Diastolic blood pressure is the bottom number, measures the force your heart exerts on the walls of your arteries in between beats. The range is between 60 and less than 80.
 3. **Poverty index:**
 4. **Race:**
 5. **Red blood cells:**
 6. **Sedimentation rate:**
 7. **Serum Albumin:**
 8. **Serum Cholesterol:**
 9. **Serum Iron:**
 10. **Serum Magnesium:**
 11. **Serum Protein:**
 12. **Sex:**
 13. **Systolic BP:** Systolic blood pressure is the top number, measures the force your heart exerts on the walls of your arteries each time it beats. The range is between 90 and less than 120.
 14. **TIBC:**
 15. **TS:**
 16. **White blood cells:**
 17. **BMI:**
 18. **Pulse pressure:**
 19. **Patient outcome:** Whether patient died within 10 years or not!

## Evaluation metric for prognostic models
The basic idea behind evaluating a prognostic model is to see how well it performs on pairs of patients. Here we're looking at death within 10 years. So we need to know that **patient A** died within the next 10 years, but **patient B** did not. Given this information for patients A and B, let's think about the risk score that a good prognostic model would give to them. A good prognostic model should give a higher risk score to patient A then to patient B.

### C-Index:
The **c-index** measures the discriminatory power of a risk score. Intuitively, a higher **c-index** indicates that the model's prediction is in agreement with the actual outcomes (e.g. if patient died or not!) of a pair of patients.

The formula for the **c-index** is given below:

$$ \mbox{c-index} = \frac{\mbox{concordant_pairs} + 0.5 \times \mbox{risk_ties}}{\mbox{permissible_pairs}} $$

* A **permissible pair** is a pair of patients who have different outcomes.
* A **concordant pair** is a permissible pair in which the patient with the higher risk score has the worse outcome.
* A **risk tie** is a permissible pair where the patients have the same risk score.

## Architecture of ML system
The following diagram explains the whole ML lifecycle of this project:

Data Ingestion --> Data validation --> Preprocessing --> Model Building/Evaluating --> Deployment --> Monitor

## EDA Insights
 - Missing at random or not (e.g. Age variable)

## Jupyter Notebook
You can find [here](https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/notebook/risk_models-classification.ipynb) the notebook containing initial efforts to explore data and build models.

## Installation
Clone this repository:
```bash
git clone <repo url>
```
Install requirements:
```python
pip install -r requirements.txt
```
Run following command:
```python
python app.py
```

## Domain
Healthcare

## Project Structure
```bash
risk-models-for-medical-prognosis/
│
├── risk_models/
│   ├── data/
│   │   ├── NHANES_I_epidemiology.csv
│   │   
│   ├── model/
│   │   └── risk_model.pkl
│   │   
├── notebook/
├── images/
│
├── tests/
│   ├── data/
│   │   ├── test.json
│   ├── conftest.py
│   └── test_predictions.py
|
├── .gitignore
├── README.md
├── requirements.txt
└── app.py
```

## Future developments:
 - Database integration
 - Batch prediction

## Acknowledgement
This project is inspired from Coursera: [AI for Medical Prognosis](https://www.coursera.org/learn/ai-for-medical-prognosis/home/welcome) course. I have build a real-world implementation of an end-to-end machine learning system upon that idea.


## Steps

1. Combine X and y data into one file -- Done
2. Load data and validate features and its data types (reference wafer fault detection)
3. Setup a pipeline that ingest data, validate data, preprocess and clean and encoding features
4. Use shap to display on UI features contributing in positive prediction (in red) and features contributing in negative prediction (in blue).