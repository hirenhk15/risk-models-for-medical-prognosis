# Risk Models for Medical Prognosis

**Please find the user interface for the demo at https://risk-models-app.herokuapp.com/**

For quick output of the demo:
<p align="center">
  <img src="https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/images/medical_prognosis_model.gif" alt="animated" />
</p>


<h2>Table of Contents<span class="tocSkip"></span></h2>
<div class="toc"><ul class="toc-item"><li><span><a href="#Domain" data-toc-modified-id="Domain-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Domain</a></span></li><li><span><a href="#What-is-medical-prognosis?" data-toc-modified-id="What-is-medical-prognosis?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>What is medical prognosis?</a></span></li><li><span><a href="#Data-Description" data-toc-modified-id="Data-Description-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Description</a></span></li><li><span><a href="#Evaluation-metric-for-prognostic-models" data-toc-modified-id="Evaluation-metric-for-prognostic-models-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Evaluation metric for prognostic models</a></span><ul class="toc-item"><li><span><a href="#C-Index" data-toc-modified-id="C-Index-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>C-Index</a></span></li></ul></li><li><span><a href="#Architecture-of-Machine-Learning-System" data-toc-modified-id="Architecture-of-Machine-Learning-System-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Architecture of Machine Learning System</a></span></li><li><span><a href="#EDA-Insights" data-toc-modified-id="EDA-Insights-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>EDA Insights</a></span></li><li><span><a href="#Jupyter-Notebook" data-toc-modified-id="Jupyter-Notebook-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Jupyter Notebook</a></span></li><li><span><a href="#Installation" data-toc-modified-id="Installation-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Installation</a></span></li><li><span><a href="#Deployment" data-toc-modified-id="Deployment-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Deployment</a></span></li><li><span><a href="#Project-Structure" data-toc-modified-id="Project-Structure-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Project Structure</a></span></li><li><span><a href="#Acknowledgement" data-toc-modified-id="Acknowledgement-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Acknowledgement</a></span></li><li><span><a href="#Future-developments" data-toc-modified-id="Future-developments-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Future developments</a></span></li></ul></div>

## Domain
Healthcare

## What is medical prognosis?
Prognosis is a medical term that refers to predicting the risk of a future event. Here, event is a general term that captures a variety of things that can happen to an individual. Events can include outcomes such as death and other adverse events like a heart attack or a stroke, which might be risks for patients who have a specific medical condition or for the general population. 

Making prognosis is a clinically useful task for a variety of reasons:
 - First, prognosis is useful for informing patients their risk of developing an illness. For example, there are blood tests that are used to estimate the risk of developing breast and ovarian cancer. 
 - Second, prognosis are also useful for informing patients how long they can expect to survive with a certain illness. An example of this is cancer staging, which gives an estimate of the survival time for patients with that particular cancer. 

Prognosis is also useful for guiding treatment. In clinical practice, the prediction of the 10-year risk of heart attack is used to determine whether a patient should get drugs to reduce the risk. Another example is the six-month mortality risk. This is used for patients with terminal conditions that have become advanced and uncurable and is used to determine who should receive end-of-life care. 

This case study is about building such prognostic models to ***predict the 10-year risk of death of an individuals from the NHANES-I epidemiology dataset*** (for a detailed description of this dataset is given at the [CDC Website](https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx/)).

Typical prognostic modelling look like (***Credit:*** See acknowledgement):
<p align="center">
  <img src="https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/images/prognostic_model.jpg" />
</p>

From above block diagram, we can see that set of features are patient profile (includes clinical history, physical examinations and labs and imaging) and the target would be risk score computed from the features (risk equation which is linear combination of natural log of features and its coefficients i.e. factor of contribution of the feature). Here natural log is taken to make equation linear.

## Data Description

Here are the description of all the features:
 1. **Age:** Age of the patient
 2. **Diastolic BP:** Diastolic blood pressure is the bottom number, measures the force your heart exerts on the walls of your arteries in between beats. The range is between 60 and less than 80.
 3. **Poverty index:** The Human Poverty Index (HPI) was an indication of the poverty of community in a country, developed by the United Nations to complement the Human Development Index (HDI) and was first reported as part of the Human Deprivation Report in 1997.
 4. **Race:** A race is a grouping of humans based on shared physical or social qualities into categories generally viewed as distinct by society.
 5. **Red blood cells:** Red blood cells, also referred to as red cells, red blood corpuscles, haematids, erythroid cells or erythrocytes, are the most common type of blood cell and the vertebrate's principal means of delivering oxygen to the body tissues—via blood flow through the circulatory system.
 6. **Sedimentation rate:** The sedimentation rate -- or “sed rate,” for short -- is a blood test that checks for inflammation in your body. It’s one clue for your doctor that you might have a disease linked to inflammation, like arthritis or cancer, or an infection.
 7. **Serum Albumin:** Serum albumin, often referred to simply as blood albumin, is an albumin (a type of globular protein) found in vertebrate blood.
 8. **Serum Cholesterol:** Cholesterol is a type of body fat, or lipid. A serum cholesterol level is a measurement of certain elements in the blood, including the amount of high- and low-density lipoprotein cholesterol (HDL and LDL) in a person’s blood.
 9. **Serum Iron:** Serum iron is a medical laboratory test that measures the amount of circulating iron that is bound to transferrin (90%) and serum ferritin (10%). Clinicians order this laboratory test when they are concerned about iron deficiency, which can cause anemia and other problems.
 10. **Serum Magnesium:** Image result for Serum Magnesium
The normal range for serum magnesium is 1.7 to 2.3 milligrams per deciliter for people 17 years old and up, according to Mayo Medical Laboratories. The exact standards for normal results may vary depending on your: age. health. body type.
 11. **Serum Protein:** A total serum protein test measures the total amount of protein in the blood. It also measures the amounts of two major groups of proteins in the blood: albumin and globulin. Albumin is made mainly in the liver. It helps keep the blood from leaking out of blood vessels.
 12. **Sex:** Organisms of many species are specialized into male and female varieties, each known as a sex.
 13. **Systolic BP:** Systolic blood pressure is the top number, measures the force your heart exerts on the walls of your arteries each time it beats. The range is between 90 and less than 120.
 14. **TIBC:** Total iron binding capacity (TIBC) is a blood test to see if you have too much or too little iron in your blood. Iron moves through the blood attached to a protein called transferrin. This test helps your health care provider know how well that protein can carry iron in your blood.
 15. **TS:** Tuberous sclerosis (TS), or tuberous sclerosis complex (TSC), is a rare genetic condition that causes noncancerous, or benign, tumors to grow in your brain, other vital organs, and skin. Sclerosis means “hardening of tissue,” and tubers are root-shaped growths.
 16. **White blood cells:** White blood cells (WBCs), also called leukocytes or leucocytes, are the cells of the immune system that are involved in protecting the body against both infectious disease and foreign invaders. All white blood cells are produced and derived from multipotent cells in the bone marrow known as hematopoietic stem cells.
 17. **BMI:** Body mass index (BMI) is a measure of body fat based on height and weight that applies to adult men and women.
 18. **Pulse pressure:** Pulse pressure is the difference between systolic and diastolic blood pressure. It is measured in millimeters of mercury (mmHg). It represents the force that the heart generates each time it contracts. Resting blood pressure is normally approximately 120/80 mmHg, which yields a pulse pressure of approximately 40 mmHg.
 19. **Patient outcome:** Whether patient died within 10 years or not!

## Evaluation metric for prognostic models
The basic idea behind evaluating a prognostic model is to see how well it performs on pairs of patients. Here we're looking at death within 10 years. So we need to know that **patient A** died within the next 10 years, but **patient B** did not. Given this information for patients A and B, let's think about the risk score that a good prognostic model would give to them. A good prognostic model should give a higher risk score to patient A then to patient B.

### C-Index
The **c-index** measures the discriminatory power of a risk score. Intuitively, a higher **c-index** indicates that the model's prediction is in agreement with the actual outcomes (e.g. if patient died or not!) of a pair of patients.

The formula for the **c-index** is given below:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?C-index&space;=&space;\frac{{\&hash;}concordant{\&space;}pairs&space;&plus;&space;0.5&space;*&space;risk{\&space;}ties}{{\&hash;permissible{\&space;}pairs}}" title="C-index = \frac{{\#}concordant{\ }pairs + 0.5 * risk{\ }ties}{{\#permissible{\ }pairs}}" />
</p>

* A **permissible pair** is a pair of patients who have different outcomes.
* A **concordant pair** is a permissible pair in which the patient with the higher risk score has the worse outcome.
* A **risk tie** is a permissible pair where the patients have the same risk score.

## Architecture of Machine Learning System
The following diagram explains the whole Machine Learning lifecycle of this project:

<p align="center">
  <img src="https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/images/ML_system.png" />
</p>

## EDA Insights

 - Following matrix shows the correlation amongst all the features:
<p align="center"><img src="https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/images/correlation_matrix.png" /></p>

 - Analyzing missing data for Systolic BP, we see that much more data tends to be missing for patients with the age over 65. The reason could be that blood pressure was measured less frequently for old people to avoid placing additional burden on them. Hence the data is missing at random as there is however no reason to believe that the values of the missing systolic blood pressures are related to the age of the patients.
<p align="center"><img src="https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/images/age_missing.png" /></p>

 - Here are the insights from below chart:
    - The red sections on the left are features which push the model towards the final prediction in the positive direction (i.e. a higher Age increases the predicted risk).
    - The blue sections on the right are features that push the model towards the final prediction in the negative direction (if an increase in a feature leads to a lower risk, it will be shown in blue).

<p align="center"><img src="https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/images/Shap_values.png" /></p>

 - Let's see a summary plot of the SHAP values for each feature on each of the test examples. The colors indicate the value of the feature.
    - Clearly we see that being a woman (`sex = 2.0`, as opposed to men for which `sex = 1.0`) has a negative SHAP value, meaning that it reduces the risk of dying within 10 years. High age and high systolic blood pressure have positive SHAP values, and are therefore related to increased mortality. 
<p align="center"><img src="https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/images/shap1.png" /></p>

 - Features interaction can be seen using dependence plots. These plot the SHAP value for a given feature for each data point, and color the points in using the value for another feature. This lets us begin to explain the variation in SHAP value for a single value of the main feature.
    - Left plot shows the interaction between Age and Sex. We see that while Age > 50 is generally bad (positive SHAP value), being a woman generally reduces the impact of age. This makes sense since we know that women generally live longer than men.
    - Right plot shows the interaction between Poverty Index and Age. We see that the impact of poverty index drops off quickly, and for higher income individuals age begins to explain much of variation in the impact of poverty index.
<p align="center">
  <img src="https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/images/shap2.png" />
  <img src="https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/images/shap3.png" />
</p>

## Jupyter Notebook
You can find [here](https://github.com/hirenhk15/risk-models-for-medical-prognosis/blob/master/notebook/risk-models-classification.ipynb) the notebook containing initial efforts to explore data and build models.

## Installation
Clone this repository:
```bash
git clone https://github.com/hirenhk15/risk-models-for-medical-prognosis.git
```
Install requirements:
```python
pip install -r requirements.txt
```
Run following command:
```python
python app.py
```

## Deployment
 - This app is deployed in Heroku.
 - CI/CD pipeline is setup for continuous integration and continuous delivery when ever code is pushed into master branch.

## Project Structure
```bash
risk-models-for-medical-prognosis/
│
├── risk_models/
│   ├── data/
│   │   ├── NHANES_I_epidemiology.csv
│   ├── model/
│   │   └── risk_model.pkl
│   ├── data_transform.py
│   ├── data_validation.py
│   ├── logger.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── train_validation.py
│   ├── util.py
│   
├── notebook/
│   ├── risk-models-classification.ipynb
├── images/
│
├── templates/
│   ├── index.html
├── tests/
│   ├── data/
│   │   ├── test.json
│   ├── conftest.py
│   └── test_app.py
|
├── .gitignore
├── README.md
├── app.py
├── config.py
├── LICENSE
├── Procfile
├── wsgi.py
├── requirements.txt
└── schema_training.json
```

## Acknowledgement
This project is inspired from Coursera: [AI for Medical Prognosis](https://www.coursera.org/learn/ai-for-medical-prognosis/home/welcome) course. I have build a real-world implementation of an end-to-end machine learning system upon that idea. Also, I would like to thank [Krish Naik](https://www.youtube.com/channel/UCNU_lfiiWBdtULKOw6X0Dig) and [iNeuron](https://ineuron.ai/) team for sharing their knowledge via YouTube videos that helped me to build this project.

## Future developments
 - Unit test and system test integration in CI/CD Pipeline.
 - Database integration to store predictions for further analytics.
 - Batch prediction from files of medical records of patients.
 - Auto monitoring of the model performance and retrain if required!
 - SHAP graphs integration to visualize important features contribuiting towards predictions.
 - Integration of API analytics such as `flask_monitoringdashboard`.
 - Create CRM module to enable each user to train their own customized risk models using their own data.