import pandas as pd
import numpy as np
import pickle
import gcsfs
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from project.utilities import create_pipeline
from project.data_loaders import MimicExtractDataLoader
from project.configs import ModelType
from project.configs import final_config_v1, TargetType
from project.target_loader import TargetLoader
from project.utilities import load_basic_hosps

FILENAME = {TargetType.MORTALITY: 'gs://mlhc_final/models/mortality_final_model.sav',
            TargetType.PROLONGED_STAY: 'gs://mlhc_final/models/prolong_stay_final_model.sav',
            TargetType.HOSPITAL_READMISSION: 'gs://mlhc_final/models/readmission_final_model.sav'}

LOCAL_FILENAME = {TargetType.MORTALITY:
                      'mortality_final_model.sav',
                  TargetType.PROLONGED_STAY:
                      'prolong_stay_final_model.sav',
                  TargetType.HOSPITAL_READMISSION:
                      'prolong_stay_final_model.sav'}


class MLHCModel(object):
    GS_DIR = 'mlhc_final/models'
    GS_PROJECT = ''

    def __init__(self, target_type, filename):
        self.target_type = target_type
        self.model = None
        self.filename = filename

    def create_model(self, model_type, X_train, y_train, model_hyper_parameters):
        """

        :param model_type:
        :param X_train:                 (pd.DataFrame) features for train
        :param y_train:                 (pd.Series) target
        :param model_hyper_parameters:  (ModelType)
        :return:
        """
        X_train_train, X_calibrate, y_train_train, y_calibrate = train_test_split(X_train, y_train, test_size=0.2,
                                                                                  stratify=y_train)

        categorical_features = X_train.columns.intersection(MimicExtractDataLoader.CATEGORICAL_FEATURES)
        intervention_features = X_train.columns.intersection(MimicExtractDataLoader.INTERVENTION_FEATURES)
        numerical_features = X_train.columns.drop(categorical_features).drop(intervention_features)

        if model_type == ModelType.RANDOM_FOREST:
            model = BalancedRandomForestClassifier(**model_hyper_parameters)
        elif model_type == ModelType.GRADIENT_BOOSTING:
            # Using Balanced Bagging method for imbalance data + Variant of Gradient Boosting:
            model = BalancedBaggingClassifier(
                estimator=HistGradientBoostingClassifier(**model_hyper_parameters),
                n_estimators=100)
        else:
            raise ValueError("Model type provided is not supported")

        pipe = create_pipeline(model, numerical_features)
        pipe.fit(X_train_train, y_train_train)
        pipe = CalibratedClassifierCV(estimator=pipe, method='isotonic', cv='prefit')
        pipe.fit(X_calibrate, y_calibrate)
        self.model = pipe
        return pipe

    def load_model(self):
        with open(self.filename, 'rb') as f:
            self.model = pickle.load(f)
        return self.model

    def load_model_from_gs(self):
        fs = gcsfs.GCSFileSystem()
        with fs.open(self.filename, 'rb') as f:
            self.model = pickle.load(f)
        return self.model

    def save_model(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.model, f)


def create_and_save_models(client, credentials, project_id):
    train_cohort = pd.read_csv('project/initial_cohort.csv')

    data_directory = 'project/data'
    hosps = load_basic_hosps(client=client)
    for target_type in TargetType:
        # Get Model config:
        config = final_config_v1[target_type]
        np.random.seed(config['best_hyper_parameters']['random_state'])

        # Load features:
        data_loader = MimicExtractDataLoader(client=client,
                                             project=project_id, credentials=credentials,
                                             initial_cohort=train_cohort,
                                             data_directory=data_directory, point_wise_agg=True,
                                             minimal_preprocessing=False,
                                             mimic_extract_los_filtering=config['mimic_extract_los_filtering'],
                                             pre_defined_feature_group=config['pre_defined_feature_group'],
                                             extend_features=config['extend_features'],
                                             filter_in_hospital_death=config['filter_in_hospital_death'],
                                             hosps=hosps)
        X_train = data_loader.get()
        # Load target for train:
        target_loader = TargetLoader(client, target_type=target_type, data_directory=None,
                                     initial_cohort=train_cohort,
                                     mimic_extract_los_filtering=config['mimic_extract_los_filtering'],
                                     filter_in_hospital_death=config['filter_in_hospital_death'],
                                     hosps=hosps)
        y_train = target_loader.get()
        X_train, y_train = X_train.align(y_train, join='inner', axis=0)

        # Create model:
        mlhc_model = MLHCModel(target_type=target_type, filename=LOCAL_FILENAME[target_type])
        pipe = mlhc_model.create_model(config['exp_model_type'], X_train, y_train, config['best_hyper_parameters'])
        mlhc_model.save_model()
