import pandas as pd

from project.utilities import (load_basic_hosps, exclude_and_include_criteria, filter_data_according_to_cohort,
                               BLACK_LIST_SAMPLES)
from project.configs import TargetType


class TargetLoader(object):
    PROLOG_STAY_HOURS = 7 * 24
    MORT_AFTER_DIS_DAYS = 30

    def __init__(self, client, mimic_extract_los_filtering, target_type=TargetType.MORTALITY,
                 initial_cohort=None, data_directory=None, filter_in_hospital_death=False, hosps=None,
                 verbose=False):
        if initial_cohort is None and data_directory is None:
            raise ValueError("Must provide one of the following: initial_cohort or data_directory.")
        if initial_cohort is None and data_directory is not None:
            # Load initial_cohort:
            initial_cohort = pd.read_csv(f'{data_directory}/initial_cohort.csv')
        self.initial_cohort = initial_cohort
        self._client = client
        self.target_type = target_type
        self.mimic_extract_los_filtering = mimic_extract_los_filtering
        self.filter_in_hospital_death = filter_in_hospital_death
        hosps = load_basic_hosps(client=self._client) if hosps is None else hosps

        if target_type == TargetType.MORTALITY:
            hosps['target'] = TargetLoader.mortality(hosps)
        elif target_type == TargetType.PROLONGED_STAY:
            hosps['target'] = TargetLoader.prolog_stay(hosps)
        elif target_type == TargetType.HOSPITAL_READMISSION:
            hosps['target'] = TargetLoader.readmission(hosps)
        else:
            raise ValueError("Target type does not supported. Please see TargetType for available targets.")

        hosps, self.cohort = exclude_and_include_criteria(hosps, self.initial_cohort, self.mimic_extract_los_filtering,
                                                          self.filter_in_hospital_death, verbose)
        hosps = hosps.set_index(['subject_id', 'hadm_id'])
        idx = pd.IndexSlice
        hosps = hosps.loc[idx[~hosps.index.get_level_values('subject_id').isin(BLACK_LIST_SAMPLES), :], :]
        # define the target:
        self.target = hosps['target']
        # self.target = filter_data_according_to_cohort(self.target, self.cohort)

    def get(self):
        return self.target

    @staticmethod
    def readmission(data):
        def is_readmission_30(patient_data):
            if patient_data.shape[0] == 1:
                return False
            return (patient_data.iloc[1].admittime - patient_data.iloc[0].dischtime).days <= 30
            # return patient_data['admittime'].shift(-1).reset_index(drop=True)
            # - patient_data['dischtime'].reset_index(drop=True)
        first_hosp_readmission = data.sort_values('admittime').groupby('subject_id').apply(
            is_readmission_30).astype(int).to_frame(name='readmission')
        subject_id_first_hadm_id = data.sort_values('admittime').groupby('subject_id').first()['hadm_id']
        first_hosp_readmission['hadm_id'] = subject_id_first_hadm_id
        return first_hosp_readmission.set_index('hadm_id', append=True).reindex(
            data.set_index(['subject_id', 'hadm_id']).index).reset_index(drop=True)


    @staticmethod
    def prolog_stay(data):
        return (data.los_hosp_hr > TargetLoader.PROLOG_STAY_HOURS).astype(int).to_frame(name='prolog_stay')

    @staticmethod
    def mortality(data):
        in_hospital_mortality = ((~data.deathtime.isna()) & (data.deathtime >= data.admittime)
                                 & (data.deathtime <= data.dischtime)).astype(int)
        after_discharge_mortality = ((data['dod'] - data['dischtime']).apply(
            lambda x: (x.days <= TargetLoader.MORT_AFTER_DIS_DAYS) & (x.days >= 0)))
        return (after_discharge_mortality | in_hospital_mortality).astype(int).to_frame(name='mortality')
