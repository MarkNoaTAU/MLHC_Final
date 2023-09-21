"""
    Config files for experiments & final models

"""
from enum import Enum


class ModelType(Enum):
    RANDOM_FOREST = 1
    GRADIENT_BOOSTING = 2


class TargetType(Enum):
    MORTALITY = 'mortality_proba'
    PROLONGED_STAY = 'prolonged_LOS_proba'
    HOSPITAL_READMISSION = 'readmission_proba'


# Experiment Prior Knowledge:
READMISSION_SIGNIFICANT_FEATURE = ['gender', 'age', 'insurance_Self Pay', 'emergency_or_urgent_admission', 'sirs',
                                   'sapsii', 'apsiii', 'sofa', 'lods', 'oasis', 'meld', 'dnr']

READMISSION_SIGNIFICANT_FEATURE_EXTENDED = ['gender', 'age', 'insurance_Self Pay', 'emergency_or_urgent_admission',
                                            'sirs',
                                            'sapsii', 'apsiii', 'sofa', 'lods', 'oasis', 'meld', 'dnr', 'aki_42hr',
                                            'underweight',
                                            'overweight', 'high_level_of_hemoglobin', 'sodium_mean', 'heart rate_std',
                                            'glucose_std']

SEPSIS_MORTALITY_ARTICLE_FEATURE_SUBSET = ['age', 'gender', 'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other',
                                           'eth_white', 'heart rate_mean', 'heart rate_min', 'heart rate_max',
                                           'temperature_mean', 'temperature_min', 'temperature_max',
                                           'respiratory rate_mean', 'respiratory rate_min', 'respiratory rate_max',
                                           'oxygen saturation_mean', 'oxygen saturation_min', 'oxygen saturation_max',
                                           'blood urea nitrogen_mean', 'diastolic blood pressure_mean',
                                           'mean blood pressure_mean', 'red blood cell count_mean',
                                           'red blood cell count csf_mean', 'red blood cell count ascites_mean',
                                           'red blood cell count pleural_mean',
                                           'red blood cell count urine_mean', 'systolic blood pressure_mean',
                                           'white blood cell count_mean', 'white blood cell count urine_mean',
                                           'blood urea nitrogen_min', 'diastolic blood pressure_min',
                                           'mean blood pressure_min',
                                           'red blood cell count_min', 'red blood cell count csf_min',
                                           'red blood cell count ascites_min', 'red blood cell count pleural_min',
                                           'red blood cell count urine_min', 'systolic blood pressure_min',
                                           'white blood cell count_min', 'white blood cell count urine_min',
                                           'blood urea nitrogen_max', 'diastolic blood pressure_max',
                                           'mean blood pressure_max',
                                           'red blood cell count_max', 'red blood cell count csf_max',
                                           'red blood cell count ascites_max', 'red blood cell count pleural_max',
                                           'red blood cell count urine_max', 'systolic blood pressure_max',
                                           'white blood cell count_max', 'white blood cell count urine_max',
                                           'glucose_mean', 'glucose_min', 'glucose_max']

HM_BASED_MORTALITY_FEATURE_SUBSET = ['age', 'gender',
                                     'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
                                     'heart rate_min', 'heart rate_max', 'heart rate_mean',
                                     'systolic blood pressure_min', 'systolic blood pressure_mean',
                                     'systolic blood pressure_max',
                                     'diastolic blood pressure_min', 'diastolic blood pressure_mean',
                                     'diastolic blood pressure_max',
                                     'mean blood pressure_mean', 'mean blood pressure_min', 'mean blood pressure_max',
                                     'respiratory rate_mean', 'respiratory rate_min', 'respiratory rate_max',
                                     'tidal volume spontaneous_mean', 'tidal volume spontaneous_min',
                                     'tidal volume spontaneous_max',
                                     'glucose_min', 'glucose_mean', 'glucose_max',
                                     'temperature_max', 'temperature_min', 'temperature_mean', 'anion gap_mean',
                                     'albumin_mean', 'bicarbonate_mean', 'bilirubin_mean',
                                     'creatinine_mean', 'chloride_mean', 'hematocrit_mean', 'hemoglobin_mean',
                                     'lactate_mean', 'magnesium_mean',
                                     'phosphate_mean', 'platelets_mean', 'potassium_mean', 'blood urea nitrogen_mean',
                                     'prothrombin time inr_mean', 'prothrombin time pt_mean',
                                     'partial thromboplastin time_mean', 'sodium_mean', 'white blood cell count_mean',
                                     'weight_mean']


# Finals Models:

mortality_v1 = dict(pre_defined_feature_group=None,
                    mimic_extract_los_filtering=True,
                    extend_features=False,
                    filter_in_hospital_death=False,
                    exp_model_type=ModelType.GRADIENT_BOOSTING,
                    best_hyper_parameters=dict(learning_rate=0.1, max_leaf_nodes=50, max_iter=100, random_state=0),
                    model_filename='project/mortality_final_model.sav'
                    )

prolog_stay_v1 = dict(pre_defined_feature_group=None,
                      mimic_extract_los_filtering=True,
                      extend_features=True,
                      filter_in_hospital_death=False,
                      exp_model_type=ModelType.RANDOM_FOREST,
                      best_hyper_parameters=dict(max_depth=10, min_samples_split=4, max_features=15, n_estimators=100,
                                                 random_state=0),
                      model_filename='project/prolong_stay_final_model.sav'
                      )

readmission_v1 = dict(pre_defined_feature_group=READMISSION_SIGNIFICANT_FEATURE,
                      mimic_extract_los_filtering=True,
                      extend_features=True,
                      filter_in_hospital_death=True,
                      exp_model_type=ModelType.GRADIENT_BOOSTING,
                      best_hyper_parameters=dict(learning_rate=0.1, max_depth=2, l2_regularization=0.25,
                                                 random_state=0),
                      model_filename='project/readmission_final_model.sav'
                      )

final_config_v1 = {TargetType.MORTALITY: mortality_v1,
                   TargetType.PROLONGED_STAY: prolog_stay_v1,
                   TargetType.HOSPITAL_READMISSION: readmission_v1}
