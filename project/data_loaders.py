import os
import numpy as np
import pandas as pd
import h5py
import gcsfs
import abc
from google.cloud import bigquery

from project.utilities import (load_basic_hosps, exclude_and_include_criteria, filter_data_according_to_cohort,
                               load_severity_scores, load_meld, load_aki)


def load_mimic_extract_group(mimic_extract_f, group='patients'):
    """
    MIMIC-III Extract Data include the following tables:
    patients:            static demographics, static outcomes -
                         One row per (subj_id,hadm_id,icustay_id)
    vitals_labs_mean:    time-varying vitals and labs (hourly mean only) -
                         One row per (subj_id,hadm_id,icustay_id,hours_in)
    interventions:       hourly binary indicators for administered interventions -
                         One row per (subj_id,hadm_id,icustay_id,hours_in)

    :param mimic_extract_f: (GCSFileSystem) GCSFileSystem directing to the h5py containing the MIMIX-EXTRACT
        pre-processed dataset. For more information see https://github.com/MLforHealth/MIMIC_Extract
        Pre-processed Output section.
    :param group: (str) one of the following group: patients, vitals_labs_mean, interventions.
    :return: (pd.DataFrame) of the relevant group dataset from MIMIC-EXTRACT.
    """
    if group not in ['patients', 'vitals_labs_mean', 'interventions']:
        raise ValueError(
            "MIMIC-III Extract Data include the following tables: 'patients', 'vitals_labs_mean', "
            "'interventions'. Group must be one of those tables.")

    # save locally and delete the h5 file to deal with the problem that pd.read_hdf expect an os.PathLike
    # and not GCSFile. h5py is structural data, and in this case build up from 4 groups.
    # We will omit vitals_labs as it's very heavy group.
    data_h5 = h5py.File(mimic_extract_f, 'r', driver='fileobj')

    # This for example load the patient dataset from MIMIC-EXTRACT:

    f = h5py.File(f'local_copy_mimic_extract_{group}.h5', 'w')
    data_h5.copy(source=data_h5.get(group), dest=f)
    f.close()
    group_extract = pd.read_hdf(f'local_copy_mimic_extract_{group}.h5')

    # Now lets delete the local copy file
    if os.path.exists(f'local_copy_mimic_extract_{group}.h5'):
        os.remove(f'local_copy_mimic_extract_{group}.h5')

    return group_extract


class DataLoader(abc.ABC):
    PRED_WINDOW_HOURS = 42

    def __init__(self, mimic_extract_los_filtering, client,
                 project='clinical-entity-extraction', credentials=None,
                 initial_cohort=None, data_directory=None, point_wise_agg=True,
                 pre_defined_feature_group=None, minimal_preprocessing=False,
                 filter_in_hospital_death=False, extend_features=False, hosps=None, verbose=False):
        """
        :param mimic_extract_los_filtering: (bool) Whether to exclude patient that are in the hospital more than 240 hours,
                                        as MIMIC-EXTRACT does.
        :param project:          (str): your Google Cloud project. You must have MIMIC-III credential in your
                                    project to load this code.
        :param initial_cohort:
        :param data_directory:
        :param point_wise_agg:
        :param pre_defined_feature_group:
        :param minimal_preprocessing:
        """
        if initial_cohort is None and data_directory is None:
            raise ValueError("Must provide one of the following: initial_cohort or data_directory.")
        if initial_cohort is None and data_directory is not None:
            # Load initial_cohort:
            initial_cohort = pd.read_csv(f'{data_directory}/initial_cohort.csv')
        self.initial_cohort = initial_cohort
        self.data_directory = data_directory
        self.mimic_extract_los_filtering = mimic_extract_los_filtering
        self.point_wise_agg = point_wise_agg
        self.pre_defined_feature_group = pre_defined_feature_group
        self.filter_in_hospital_death = filter_in_hospital_death
        self.extend_features = extend_features
        self._project = project
        self._client = client
        self._project_id = project
        self._credentials = credentials
        hosps = load_basic_hosps(client=self._client) if hosps is None else hosps
        _, self.cohort = exclude_and_include_criteria(hosps, self.initial_cohort,
                                                      self.mimic_extract_los_filtering, self.filter_in_hospital_death,
                                                      verbose)
        self.minimal_preprocessing = minimal_preprocessing

    def get(self):
        """
            loading the feature data and filtering according to the exclusion criteria.
            If data_directory is provided, and cache = True, will try to load
            the data from pickles (if not exists, load and save).
            Will select only the relevant cohort, and aggregate the features if needed.
            If specfic pre_defined_feature_group are given, return only this subset.
        :return: (pd.DataFrame) features
        """
        # first load data as is:
        if self.data_directory is not None:
            data = self.load_from_pickle()
            if data is None:
                # AKA there was no pickle to load...
                data = self.load_all_data()
                self.save_to_pickle(data)
        else:
            data = self.load_all_data()

        # select cohort
        data = filter_data_according_to_cohort(data, self.cohort)

        # filter only first 42 hours!
        data = self._filter_hours(data)

        # aggregate:
        if self.point_wise_agg:
            data = self._point_wise_aggregation(data)

        # data can be tuple of pd.DataFrame, then aggregate to a single df.
        if isinstance(data, tuple):
            data = self._allign_features(data)

        # pre-process categorical features:
        if not self.minimal_preprocessing:
            data = self._pre_process_categorical_features(data)

        # additional features:
        if self.extend_features:
            data = self._extract_additional_features(data)

        # select pre-defined features:
        if self.pre_defined_feature_group is not None:
            data = data.loc[:, self.pre_defined_feature_group]

        return data

    @abc.abstractmethod
    def _extract_additional_features(self, data):
        """ Add additional features, on top of the default setting """

    @staticmethod
    def _one_hot_ethnicity(data):
        ethnicities_types = ['eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white']
        data.ethnicity = data.ethnicity.str.lower()
        data.loc[(data.ethnicity.str.contains('^white')), 'ethnicity'] = 'white'
        data.loc[(data.ethnicity.str.contains('^black')), 'ethnicity'] = 'black'
        data.loc[
            (data.ethnicity.str.contains('^hisp')) | (data.ethnicity.str.contains('^latin')), 'ethnicity'] = 'hispanic'
        data.loc[(data.ethnicity.str.contains('^asia')), 'ethnicity'] = 'asian'
        data.loc[
            ~(data.ethnicity.str.contains('|'.join(['white', 'black', 'hispanic', 'asian']))), 'ethnicity'] = 'other'
        ethnicities = pd.get_dummies(data['ethnicity'],
                                     prefix='eth').reindex(columns=ethnicities_types).fillna(0).astype(int)
        data = pd.concat([data, ethnicities], axis=1)
        data = data.drop(columns='ethnicity')
        return data

    def _pre_process_categorical_features(self, data):
        # Gender to binary:
        if 'gender' in data.columns:
            data['gender'] = np.where(data['gender'] == "M", 1, 0)

        # Ethnicity - one hot encoding
        if 'ethnicity' in data.columns:
            data = DataLoader._one_hot_ethnicity(data)
        return data

    @abc.abstractmethod
    def _allign_features(self, data):
        """ data may be a pd.DataFrame or a tuple. After this call it will be a tuple. """

    @abc.abstractmethod
    def _filter_hours(self, data):
        """ Use only data from the first 42 hours of admission. """

    @abc.abstractmethod
    def _point_wise_aggregation(self, data):
        """ aggregate time-series measurements (such as labs and vitals) in data to a single point-wise value."""

    @abc.abstractmethod
    def load_all_data(self):
        """ load features. """

    @abc.abstractmethod
    def load_from_pickle(self):
        pass

    @abc.abstractmethod
    def save_to_pickle(self, data):
        pass


class MimicExtractDataLoader(DataLoader):
    F_PATH = 'gs://mimic_extract/all_hourly_data.h5'
    CATEGORICAL_FEATURES = ['gender', 'dnr', 'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
                            'insurance_Government', 'insurance_Medicaid', 'insurance_Medicare', 'insurance_Private',
                            'insurance_Self Pay', 'first_careunit_CCU', 'first_careunit_CSRU', 'first_careunit_MICU',
                            'first_careunit_SICU', 'first_careunit_TSICU', 'admission_type_ELECTIVE',
                            'admission_type_EMERGENCY', 'admission_type_URGENT', 'emergency_or_urgent_admission',
                            'high_level_of_hemoglobin', 'underweight', 'overweight', 'aki_42hr']
    INTERVENTION_FEATURES = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel',
                             'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus',
                             'crystalloid_bolus', 'nivdurations']
    SEVERITY_SCORE_FEATURES = ['sirs', 'sapsii', 'apsiii', 'sofa', 'lods', 'oasis']

    def __init__(self, mimic_extract_los_filtering, client,
                 project='clinical-entity-extraction', credentials=None,
                 initial_cohort=None, data_directory=None, point_wise_agg=True,
                 pre_defined_feature_group=None, minimal_preprocessing=False,
                 filter_in_hospital_death=False, extend_features=False, hosps=None):
        """
            Variable limits, Unit conversion, Union of concepts - All was done for us as part of MIMIC-EXTRACT data
            loading
        """

        super().__init__(mimic_extract_los_filtering=mimic_extract_los_filtering, client=client, project=project,
                         credentials=credentials, initial_cohort=initial_cohort, data_directory=data_directory,
                         point_wise_agg=point_wise_agg, pre_defined_feature_group=pre_defined_feature_group,
                         minimal_preprocessing=minimal_preprocessing,
                         filter_in_hospital_death=filter_in_hospital_death, extend_features=extend_features,
                         hosps=hosps)
        self.icustay_id = None

    @staticmethod
    def _one_hot_insurance(data):
        """
            In MIMIC-III the INSURANCE, LANGUAGE, RELIGION, MARITAL_STATUS, ETHNICITY columns describe patient
            demographics. MIMIC-EXTRACT available: ETHNICITY & INSURANCE.
        """
        # When loading small subset of the data, some of the categories might not be there. To validate all exists:

        insurance_types = ['insurance_Government', 'insurance_Medicaid', 'insurance_Medicare', 'insurance_Private',
                           'insurance_Self Pay']
        insurance = pd.get_dummies(data['insurance'], prefix='insurance').reindex(columns=insurance_types
                                                                                  ).fillna(0).astype(int)
        return pd.concat([data, insurance], axis=1).drop(columns='insurance')

    @staticmethod
    def _one_hot_first_care_unit(data):
        """
            First care unit might also indicate something about severity.
        """
        first_care_units_types = ['first_careunit_CCU', 'first_careunit_CSRU', 'first_careunit_MICU',
                                  'first_careunit_SICU', 'first_careunit_TSICU']
        first_care_units = pd.get_dummies(data['first_careunit'], prefix='first_careunit').reindex(
            columns=first_care_units_types).fillna(0).astype(int)

        return pd.concat([data, first_care_units], axis=1).drop(columns='first_careunit')

    @staticmethod
    def _ont_hot_admission_type(data):
        """
            Admission type might also indicate somthing about the severity of the
            patient illness.
        """
        admission_type = ['admission_type_ELECTIVE', 'admission_type_EMERGENCY', 'admission_type_URGENT']
        addmisions_types = pd.get_dummies(data['admission_type'],
                                          prefix='admission_type').reindex(columns=admission_type).fillna(0).astype(int)
        return pd.concat([data, addmisions_types], axis=1).drop(columns='admission_type')

    @staticmethod
    def _dnr(data):
        """
    process DNR- Do not Resuscitate:
    might significantly relate to mortality, Use only if available at prediction time-window.
    if the first_charttime is in the first 42 hours of admission & dnr is 1 -
    its 1 else nan, zero or later timestamps its 0.
    """
        if ('dnr' in data.columns) & ('dnr_first_charttime' in data.columns):
            data['dnr'] = ((data['dnr_first_charttime'] - data['admittime']).apply(
                lambda x: x.total_seconds() // 3600) <= DataLoader.PRED_WINDOW_HOURS) & data['dnr']
            data = data.drop(columns=['dnr_first_charttime', 'dnr_first'])
        return data

    def _pre_process_categorical_features(self, data):
        data = super()._pre_process_categorical_features(data)
        data = MimicExtractDataLoader._dnr(data)
        data = MimicExtractDataLoader._one_hot_insurance(data)
        data = MimicExtractDataLoader._one_hot_first_care_unit(data)
        data = MimicExtractDataLoader._ont_hot_admission_type(data)
        # drop features:
        data = data.drop(
            columns=['admittime', 'dischtime', 'deathtime', 'discharge_location', 'mort_icu', 'mort_hosp',
                     'hospital_expire_flag',
                     'hospstay_seq', 'readmission_30', 'max_hours', 'intime', 'outtime', 'los_icu'])
        # CMO = comfort measures only; COM is a care plan that includes physician orders that address patient's
        # potential bodily symptoms of discomfort that may be implemented when curative treatment has been stopped
        # and death is expected -
        # Same as DNR if given in the first 42 hours of addmition, might be very relevant!!!
        # But CMO in MIMIC-EXTRACT does not contain a timestamp information, and therefore can
        # not be used in our format. (Might be somehow available, didn't inspect it further.)
        data = data.drop(columns=['cmo_first', 'cmo_last', 'cmo'])

        # diagnosis_at_admission -  is free-text diagnosis for the patient on hospital admission
        # According to MIMIC documentation: While this field can provide information about the status of
        # a patient on hospital admission, it is not recommended to use it to stratify patients.
        data = data.drop(columns='diagnosis_at_admission')

        # Full code is some medical simulation tool or something, I couldn't find good documentation and clear
        # usage for this feature - so currently drop it.
        data = data.drop(columns=['fullcode_first', 'fullcode'])
        return data

    def _allign_features(self, data):
        patients_extract, vitals_labs_mean_extract, interventions_extract = data
        patients_extract = patients_extract.droplevel('icustay_id')
        vitals_labs_mean_extract = vitals_labs_mean_extract.droplevel('icustay_id')
        interventions_extract = interventions_extract.droplevel('icustay_id')
        data = patients_extract.join(vitals_labs_mean_extract).join(interventions_extract)
        return data

    def _filter_hours(self, data):
        patients_extract, vitals_labs_mean_extract, interventions_extract = data
        # I use fixed number instead of Magic Number due to this bug: https://github.com/pandas-dev/pandas/issues/54449
        vitals_labs_mean_extract = vitals_labs_mean_extract.query('hours_in <= 42')
        interventions_extract = interventions_extract.query('hours_in <= 42')
        return patients_extract, vitals_labs_mean_extract, interventions_extract

    def _point_wise_aggregation(self, data):
        """ aggregate time-series measurements (such as labs and vitals) in data to a single point-wise value."""
        patients_extract, vitals_labs_mean_extract, interventions_extract = data
        # for each feature in the vitals_labs_mean_extract, take the min, max and mean across hours.
        # for each feature in the vitals_labs_mean_extract, take the min, max and mean across hours.
        vitals_labs_mean_extract = vitals_labs_mean_extract.droplevel(axis=1, level='Aggregation Function')
        vitals_labs_mean_extract.columns.names = ['feature_name']
        if self.extend_features:
            vitals_labs_mean_extract = pd.concat(
                [vitals_labs_mean_extract.groupby(['subject_id', 'hadm_id', 'icustay_id']).mean(),
                 vitals_labs_mean_extract.groupby(['subject_id', 'hadm_id', 'icustay_id']).min(),
                 vitals_labs_mean_extract.groupby(['subject_id', 'hadm_id', 'icustay_id']).max(),
                 vitals_labs_mean_extract.groupby(['subject_id', 'hadm_id', 'icustay_id']).std()], axis=1,
                keys=['mean', 'min', 'max', 'std'])
            idx = pd.IndexSlice
            spread = vitals_labs_mean_extract.loc[:, idx['max', :]].droplevel(0, axis=1) - vitals_labs_mean_extract.loc[
                                                                                           :, idx['min', :]].droplevel(
                0, axis=1)
            spread.columns = pd.MultiIndex.from_product([['spread'], spread.columns])
            vitals_labs_mean_extract = pd.concat([vitals_labs_mean_extract, spread])
        else:
            vitals_labs_mean_extract = pd.concat(
                [vitals_labs_mean_extract.groupby(['subject_id', 'hadm_id', 'icustay_id']).mean(),
                 vitals_labs_mean_extract.groupby(['subject_id', 'hadm_id', 'icustay_id']).min(),
                 vitals_labs_mean_extract.groupby(['subject_id', 'hadm_id', 'icustay_id']).max()], axis=1,
                keys=['mean', 'min', 'max'])
        vitals_labs_mean_extract = vitals_labs_mean_extract.swaplevel(i=0, j=1, axis=1)
        vitals_labs_mean_extract.columns = ['_'.join(col) for col in vitals_labs_mean_extract.columns.values]

        # for interventions_extract, take the max across the hours (whether this intervention happened in
        # this time-window) we could also look on number of hours of intervention,
        # or the max-time-window the intervention acure - but we chose to keep it simple.
        interventions_extract = interventions_extract.groupby(['subject_id', 'hadm_id', 'icustay_id']).max()
        return patients_extract, vitals_labs_mean_extract, interventions_extract

    def _extract_additional_features(self, data):
        # severity score:
        severity_scores = load_severity_scores(self._client)
        data = data.join(severity_scores.set_index(['hadm_id', 'subject_id']), how='inner')
        # BMI:
        data['bmi'] = (data['weight_mean'] / ((data['height_mean'] / 100) ** 2)).replace(
            [np.inf, -np.inf], np.nan).clip(upper=300, lower=10)
        # MELD:
        meld = load_meld(self._client)
        data = data.join(meld.set_index(['hadm_id', 'subject_id']), how='inner')
        # Yes/ no emergency_or_urgent_admission
        data['emergency_or_urgent_admission'] = data['admission_type_EMERGENCY'] | data['admission_type_URGENT']
        # High hemoglobin (> 12)
        data['high_level_of_hemoglobin'] = data['hemoglobin_mean'] > 12
        # underweight
        data['underweight'] = data['bmi'] < 18
        # overweight
        data['overweight'] = data['bmi'] > 25
        # AKI:
        data['aki_42hr'] = load_aki(self._client, self.icustay_id)

        data = data[~data.index.duplicated(keep='first')]

        return data

    def load_all_data(self):
        """ load features. """
        fs = gcsfs.GCSFileSystem(project=self._project_id,
                                 token=self._credentials)
        mimic_extract_f = fs.open(MimicExtractDataLoader.F_PATH)
        patients_extract = load_mimic_extract_group(mimic_extract_f, group='patients')
        vitals_labs_mean_extract = load_mimic_extract_group(mimic_extract_f, group='vitals_labs_mean')
        interventions_extract = load_mimic_extract_group(mimic_extract_f, group='interventions')
        icustay_id = patients_extract.reset_index(-1)['icustay_id']
        self.icustay_id = pd.DataFrame({'subject_id': icustay_id.index.get_level_values(0),
                                        'hadm_id': icustay_id.index.get_level_values(1)}, index=icustay_id)
        return patients_extract, vitals_labs_mean_extract, interventions_extract

    def load_from_pickle(self):
        """
        If files exists - load them, else return None
      """
        if os.path.isfile(f'{self.data_directory}/local_copy_mimic_extract_patients.pkl') & os.path.isfile(
                f'{self.data_directory}/local_copy_mimic_extract_vitals_labs_mean.pkl') & os.path.isfile(
            f'{self.data_directory}/local_copy_mimic_extract_interventions.pkl'):
            patients_extract = pd.read_pickle(f'{self.data_directory}/local_copy_mimic_extract_patients.pkl')
            vitals_labs_mean_extract = pd.read_pickle(
                f'{self.data_directory}/local_copy_mimic_extract_vitals_labs_mean.pkl')
            interventions_extract = pd.read_pickle(
                f'{self.data_directory}/local_copy_mimic_extract_interventions.pkl')

            icustay_id = patients_extract.reset_index(-1)['icustay_id']
            self.icustay_id = pd.DataFrame({'subject_id': icustay_id.index.get_level_values(0),
                                            'hadm_id': icustay_id.index.get_level_values(1)}, index=icustay_id)
            return patients_extract, vitals_labs_mean_extract, interventions_extract
        else:
            return None

    def save_to_pickle(self, data):
        patients_extract, vitals_labs_mean_extract, interventions_extract = data
        patients_extract.to_pickle(f'{self.data_directory}/local_copy_mimic_extract_patients.pkl')
        vitals_labs_mean_extract.to_pickle(f'{self.data_directory}/local_copy_mimic_extract_vitals_labs_mean.pkl')
        interventions_extract.to_pickle(f'{self.data_directory}/local_copy_mimic_extract_interventions.pkl')


class HomeWorkDataLoader(DataLoader):
    VITAL_QUERY = \
        """--sql
        -- Vital signs include heart rate, blood pressure, respiration rate, and temperature

          SELECT chartevents.subject_id ,chartevents.hadm_id ,chartevents.charttime
          , chartevents.itemid, chartevents.valuenum
          , admissions.admittime
          FROM `physionet-data.mimiciii_clinical.chartevents` chartevents
          INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
          ON chartevents.subject_id = admissions.subject_id
          AND chartevents.hadm_id = admissions.hadm_id
            AND chartevents.charttime between (admissions.admittime) AND (admissions.admittime + interval '42' hour)
          AND itemid in UNNEST(@itemids)
          -- exclude rows marked as error
          AND chartevents.error IS DISTINCT FROM 1
          """

    LABS_QUERY = \
        """--sql
          SELECT labevents.subject_id ,labevents.hadm_id ,labevents.charttime
          , labevents.itemid, labevents.valuenum
          , admissions.admittime
          FROM `physionet-data.mimiciii_clinical.labevents` labevents
            INNER JOIN `physionet-data.mimiciii_clinical.admissions` admissions
            ON labevents.subject_id = admissions.subject_id
            AND labevents.hadm_id = admissions.hadm_id
              AND labevents.charttime between (admissions.admittime) AND (admissions.admittime + interval '42' hour)
            AND itemid in UNNEST(@itemids)
        """

    def __init__(self, mimic_extract_los_filtering, client,
                 project='clinical-entity-extraction', credentials=None,
                 initial_cohort=None, data_directory=None, point_wise_agg=True,
                 pre_defined_feature_group=None, minimal_preprocessing=False,
                 filter_in_hospital_death=False, extend_features=False, hosps=None):
        super().__init__(mimic_extract_los_filtering=mimic_extract_los_filtering, client=client, project=project,
                         credentials=credentials, initial_cohort=initial_cohort, data_directory=data_directory,
                         point_wise_agg=point_wise_agg, pre_defined_feature_group=pre_defined_feature_group,
                         minimal_preprocessing=minimal_preprocessing,
                         filter_in_hospital_death=filter_in_hospital_death, extend_features=extend_features,
                         hosps=hosps)
        self.labs_metadata = pd.read_csv(f'{self.data_directory}/labs_metadata.csv')
        self.vital_meta_data = pd.read_csv(f'{self.data_directory}/vital_metadata.csv')

    def _extract_additional_features(self, data):
        """ Add additional features, on top of the default setting """
        return data

    def _pre_process_categorical_features(self, data):
        data = super()._pre_process_categorical_features(data)
        # drop features:
        data = data.drop(
            columns=['admittime', 'dischtime', 'deathtime', 'los_hosp_hr', 'dob', 'dod'])
        return data

    def _allign_features(self, data):
        """ data may be a pd.DataFrame or a tuple. After this call it will be a tuple. """
        hosps, vits_and_labs = data
        return hosps.set_index(['subject_id', 'hadm_id']).join(vits_and_labs.set_index(['subject_id', 'hadm_id']))

    def _filter_hours(self, data):
        """ Use only data from the first 42 hours of admission. """
        # we already filter labs and vitals measurements to the first 42 hours in the BigQuery query, but let's verify:

        hosps, labs, vitals = data

        def cal_hours_in(df):
            df['hours_in'] = (df.charttime - df.admittime).apply(lambda x: x.total_seconds() // 3600).astype(int)
            return df

        vitals = cal_hours_in(vitals)
        labs = cal_hours_in(labs)
        assert (vitals.hours_in <= 42).all()
        assert (labs.hours_in <= 42).all()
        return data

    def _point_wise_aggregation(self, data):
        """ aggregate time-series measurements (such as labs and vitals) in data to a single point-wise value."""
        hosps, labs, vitals = data
        vits_and_labs = pd.concat([labs, vitals])
        vits_and_labs = pd.pivot_table(vits_and_labs, values='valuenum',
                                       index=['subject_id', 'hadm_id'],
                                       columns=['feature name'], aggfunc=[min, max, np.mean]
                                       )
        vits_and_labs.columns = ['_'.join(col) for col in vits_and_labs.columns.values]
        vits_and_labs = vits_and_labs.reset_index()
        return hosps, vits_and_labs

    def _load_and_filter_based_on_metadata(self, query, metadata, hosps):
        """ load query and filter the features based on metadata and hosps admission ids."""
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("itemids", "INTEGER", metadata['itemid'].tolist()),
            ]
        )
        loaded_data = self._client.query(query, job_config=job_config).result().to_dataframe().rename(str.lower,
                                                                                                      axis='columns')
        # filter invalid measurements:
        loaded_data = loaded_data[loaded_data['hadm_id'].isin(hosps['hadm_id'])]
        loaded_data = pd.merge(loaded_data, metadata, on='itemid')
        loaded_data = loaded_data[loaded_data['valuenum'].between(
            loaded_data['min'], loaded_data['max'], inclusive='both')]
        return loaded_data

    def load_all_data(self):
        """ load features. """
        hosps = load_basic_hosps(client=self._client)
        labs = self._load_and_filter_based_on_metadata(HomeWorkDataLoader.LABS_QUERY, self.labs_metadata, hosps)
        vitals = self._load_and_filter_based_on_metadata(HomeWorkDataLoader.VITAL_QUERY, self.vital_meta_data, hosps)
        # unit conversion:
        vitals.loc[(vitals['feature name'] == 'TempF'), 'valuenum'] = (vitals[vitals['feature name'] == 'TempF'][
                                                                           'valuenum'] - 32) / 1.8
        vitals.loc[vitals['feature name'] == 'TempF', 'feature name'] = 'TempC'
        return hosps, labs, vitals

    def load_from_pickle(self):
        return None

    def save_to_pickle(self, data):
        pass
