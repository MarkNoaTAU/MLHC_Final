# Machine Learning for Health Care Final Project TLV university 2023

## Project description 
This library contains research code for prediction mortality, prolong stay and hospital readmission using
MIMIC-III dataset. 
This is final work for *"Machine Learning for Health Care"* course by Omer Noy. 

For more details about this project, please review [this project overview doc](https://github.com/MarkNoaTAU/MLHC_Final/blob/main/ML%20for%20healthcare%20-%20project%20overview.pdf).


Writers: Noa Mark.

## Installation:
To run the code please first install requirements.py

## Running:
The course had a test-file to run the project. 
Note, as we load data using google storage alongside the use of big query, we also need credential and project id 
to access MIMIC-III google storage data. 

The test-notebook requires the following changes:

1. Add the following lines after calling to auth.authenticate_user():
import google.auth
credentials, project_id = google.auth.default()

2. Instand of running the following line:
predction_df = run_pipeline_on_unseen_data(test['subject_id'].to_list())
Run:
predction_df = run_pipeline_on_unseen_data(test['subject_id'].to_list(), client,  credentials, project_id)


## API 
*   **Loading the data** - *data_loaders.py*:

    Contain the different data loaders, where DataLoader provide an abstract API and common
    functionalities, and MimicExtractDataLoader and HomeWorkDataLoader are specific data loaders implementations.


*  **Loading the target** - *target_loader.py*:
    Contain TargetLoader the loads the target. Target is set according to TargetType in configs, and can be 
    one of the following: mortality, prolong stay or readmission. 
    

*  **Evaluation** - *evaluation.py*:

    Contain functions to aid evaluating the models, such as:
   *  Plot ROC & PR curves with bootstrap confidence evaluation 
   * Calculate and print different evaluation matrices: acc, balance acc, f1 score, PR80, AP. 
     As well as print confusion matrix of the given data.
   * Feature importance - plotting and calculating. Using SHAP values or build-in Random forest feature importance.
   


*  **Datat transformers** - *data_transformers.py*:

    Contain build in data transformers inheriting from sklearn BaseEstimator, TransformerMixin.
   * AgeGenderImputer - imputing numerical values based on age-gender group.
   * SimpleImputerWrapper -  A dummy custom wrapper to save the columns names info.
   * StandardScalerCustomWrapper - A dummy custom wrapper to handle working with AgeGenderImputer and ColumnTransformer,
    without scaling gender.


*  **Experiments & Setting** - 

   *experiment_runner.py*: 
    Contain helper function for running the research experiments such as hyperparameter tunning and feature selection.
   (The experiment was running on Google Colab notebooks.)
    
    *configs.py*:
    Contain useful configuration such as the final models configurations, as well as some prior-knowledge constants.
    For example, following knowledge from articles and feature analysis groups of relevant features where created
   (see READMISSION_SIGNIFICANT_FEATURE, SEPSIS_MORTALITY_ARTICLE_FEATURE_SUBSET etc.)

