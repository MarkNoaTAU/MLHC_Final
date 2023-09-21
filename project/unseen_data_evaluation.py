import pandas as pd
import numpy as np

from project.data_loaders import MimicExtractDataLoader
from project.configs import final_config_v1, TargetType
from project.utilities import load_basic_hosps
from project.train import MLHCModel, FILENAME


def run_pipeline_on_unseen_data(subject_ids, client, credentials, project_id):
    """
    Run your full pipeline, from data loading to prediction.

    :param subject_ids: A list of subject IDs of an unseen test set.
    :type subject_ids: List[int]

    :param client: A BigQuery client object for accessing the MIMIC-III dataset.
    :type client: google.cloud.bigquery.client.Client

    :param credentials: A google auth credentials for accessing the MIMIC-III dataset
    :param project_id: A google project id with credentials for accessing the MIMIC-III dataset

    :return: DataFrame with the following columns:
                - subject_id: Subject IDs, which in some cases can be different due to your analysis.
                - mortality_proba: Prediction probabilities for mortality.
                - prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
                - readmission_proba: Prediction probabilities for readmission.
    :rtype: pandas.DataFrame
    """
    test_cohort = pd.DataFrame({'subject_id': subject_ids})
    data_directory = 'project/data'
    hosps = load_basic_hosps(client=client)
    final_pred = {}
    for target_type in TargetType:
        # Get Model config:
        config = final_config_v1[target_type]
        np.random.seed(config['best_hyper_parameters']['random_state'])

        # Load features:
        data_loader = MimicExtractDataLoader(client=client,
                                             project=project_id, credentials=credentials,
                                             initial_cohort=test_cohort,
                                             data_directory=data_directory, point_wise_agg=True,
                                             minimal_preprocessing=False,
                                             mimic_extract_los_filtering=config['mimic_extract_los_filtering'],
                                             pre_defined_feature_group=config['pre_defined_feature_group'],
                                             extend_features=config['extend_features'],
                                             filter_in_hospital_death=config['filter_in_hospital_death'],
                                             hosps=pd.DataFrame.copy(hosps))
        X_test = data_loader.get()

        # Load model:
        mlhc_model = MLHCModel(target_type=target_type, filename=FILENAME[target_type])
        pipe = mlhc_model.load_model_from_gs()

        # Predict probabilities:
        pred = pipe.predict_proba(X_test)
        final_pred[target_type.value] = pd.DataFrame({target_type.value: pred[:, 1],
                                                      'subject_id': X_test.index.get_level_values('subject_id')}
                                                     ).set_index('subject_id')

    # Merge to final DataFrame.
    final_pred = final_pred['mortality_proba'].join(final_pred['prolonged_LOS_proba'], how='outer').join(
        final_pred['readmission_proba'], how='outer').reset_index(drop=False)
    return final_pred
