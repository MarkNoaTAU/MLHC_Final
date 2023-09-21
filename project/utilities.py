import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from project.data_transformers import AgeGenderImputer, SimpleImputerWrapper, StandardScalerCustomWrapper

# There are a few samples (<10) where the second readmission was during the first one. We will drop this samples:
BLACK_LIST_SAMPLES = [9896, 9998, 13890, 14219, 17964, 23843, 26690, 29175, 30940]


def create_pipeline(model, numerical_features):
    pre_pipe = Pipeline(steps=[("imputer", AgeGenderImputer()),
                               ('backupImputer', SimpleImputerWrapper()),
                               ('scaler', StandardScalerCustomWrapper())])
    preprocessor = ColumnTransformer(transformers=[
        ("impute_and_scale", pre_pipe, numerical_features.append(pd.Index(['gender'])))], remainder='passthrough')

    # ft = preprocessor.fit_transform(X_train_train)

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipe


def load_basic_hosps(client):
    """
    Load basic static patient data from MIMIC-III
    :param client: (google.cloud.bigquery.client.Client) A BigQuery client object for accessing the MIMIC-III dataset.
    :return: (pd.DataFrame) Contain basic static information such as subject_id, age, ethnicity etc.
    """
    hospquery = \
        """
    SELECT admissions.subject_id, admissions.hadm_id
    , admissions.admittime, admissions.dischtime, admissions.deathtime
    , admissions.ethnicity
    , patients.gender, patients.dob, patients.dod
    FROM `physionet-data.mimiciii_clinical.admissions` admissions
    INNER JOIN `physionet-data.mimiciii_clinical.patients` patients
      ON admissions.subject_id = patients.subject_id
    WHERE admissions.has_chartevents_data = 1
    ORDER BY admissions.subject_id, admissions.hadm_id, admissions.admittime;
    """
    hosps = client.query(hospquery).result().to_dataframe().rename(str.lower, axis='columns')

    def age(admittime, dob):
        if admittime < dob:
            return 0
        return admittime.year - dob.year - ((admittime.month, admittime.day) < (dob.month, dob.day))

    hosps['age'] = hosps.apply(lambda row: age(row['admittime'], row['dob']), axis=1)
    # hosps['los_hosp_hr'] = (hosps.dischtime - hosps.admittime).astype('timedelta64[h]')
    hosps['los_hosp_hr'] = (hosps.dischtime - hosps.admittime).dt.floor("H").astype('timedelta64[s]') / 3600
    return hosps


def exclude_and_include_criteria(hosps, initial_cohort, mimic_extract_los_filtering=True,
                                 filter_in_hospital_death=False, verbose=False):
    """
    Filter the hosps DataFrame according to the following exclusion criteria:

    1. Only patient with at least 48 hours of hospitalization data.
    2. only patient with age [18,89] (Exclude younger or older)
    3. Only first admission.
    4. Exclude patient that died in the first 48 hours
    5. If mimic_extract_los_filtering is True exclude patient with LOS larger than 240 hours, else, ignore.

    Return tuple : filter_hosps, cohort_subject_id
    Where cohort_subject_id is a filtered dataframe of the initial_cohort that also meet our exclusion criteria.

    :param hosps:                       (pd.DataFrame) Contain admissions data about the patient, must contain
                                        admittime, dischtime, age, dod and subject_id.
    :param initial_cohort:               (pd.DataFrame) Contain subject_id subsets.
                                        The subset of cohort allowed to use in this project.
    :param mimic_extract_los_filtering: (bool) Whether to exclude patient that are in the hospital more than 240 hours,
                                        as MIMIC-EXTRACT does.
    :param filter_in_hospital_death:    (bool) Filter in hospital death.
    :return:                            (pd.DataFrame, pd.DataFrame) a tuple.
                                        The hosps dataframe filtered to according to the excluding and
                                        including criteria, and pd.DataFrame of the selected cohort subject_id.
    """
    # So on top of the initial_cohort lets filter according those criteria:
    hosps_initial_cohort = hosps[hosps.subject_id.isin(initial_cohort.subject_id)]
    if verbose:
        print(f"0. Include only inital_cohort: N={hosps_initial_cohort.shape[0]}")
    # Only first admission:
    hosps_initial_cohort = hosps_initial_cohort.sort_values('admittime').groupby('subject_id').first().reset_index()
    if verbose:
        print(f"1. Include only first addmision: N={hosps_initial_cohort.shape[0]}")

    # Exclude by age:
    hosps_initial_cohort = hosps_initial_cohort.query('age >= 18 & age <= 89')
    if verbose:
        print(f"2. Include only ages [18,89]: N={hosps_initial_cohort.shape[0]}")

    # Eclude by LOS: at least 48 hours (and possibly no more than 240):
    if mimic_extract_los_filtering:
        hosps_initial_cohort = hosps_initial_cohort.query('los_hosp_hr >= 48 & los_hosp_hr <= 240')
        if verbose:
            print(f"3. Include only LOS [48,240]: N={hosps_initial_cohort.shape[0]}")
    else:
        hosps_initial_cohort = hosps_initial_cohort.query('los_hosp_hr >= 48')
        if verbose:
            print(f"3. Include only LOS > 48: N={hosps_initial_cohort.shape[0]}")

    # Exclude by in-hospital mortality - died in the first 48 hours:
    min_target_onset = 48
    hosps_initial_cohort = hosps_initial_cohort[~(((~ hosps_initial_cohort.dod.isna())
                                                   & (hosps_initial_cohort.dod >= hosps_initial_cohort.admittime)
                                                   & (hosps_initial_cohort.dod <= hosps_initial_cohort.dischtime))
                                                  & (hosps_initial_cohort.los_hosp_hr < min_target_onset))]
    if verbose:
        print(f"4. Exclude patient that died in the first 48 hours: N={hosps_initial_cohort.shape[0]}")

    if filter_in_hospital_death:
        hosps_initial_cohort = hosps_initial_cohort[
            ~((~hosps_initial_cohort.deathtime.isna())
              & (hosps_initial_cohort.deathtime <= hosps_initial_cohort.dischtime)
              & (hosps_initial_cohort.deathtime >= hosps_initial_cohort.admittime))]
        if verbose:
            print(f"5. Exclude in hospital death: N={hosps_initial_cohort.shape[0]}")

    return hosps_initial_cohort, hosps_initial_cohort.subject_id.reset_index(drop=True).to_frame()


def filter_data_according_to_cohort(data, cohort_subject_id):
    """

    :param data:                (pd.DataFrame) any dataset that contain subject_id as index/columns.
    :param cohort_subject_id:   (pd.DataFrame) that contain a subject of subject_id that define the cohort.
    :return:                    (pd.DataFrame) filtered_data, a subset of the data that aligned with the cohort.
    """

    def _filter_df(_data, _cohort_subject_id):
        if 'subject_id' in _data.columns:
            filtered_data = _data[_data.subject_id.isin(_cohort_subject_id.subject_id)]
        elif 'subject_id' in _data.index.names:
            filtered_data = _data[_data.index.get_level_values('subject_id').isin(_cohort_subject_id.subject_id)]
        else:
            raise ValueError("data must contain subject_id information whether in columns or in index levels.")
        return filtered_data

    if isinstance(data, pd.DataFrame):
        data = _filter_df(data, cohort_subject_id)
    elif isinstance(data, tuple):
        data = tuple([_filter_df(data_t, cohort_subject_id) for data_t in data])
    else:
        raise ValueError("Data must be pd.DataFrame or tuple of pd.DataFrame")
    return data


def load_severity_scores(client):
    """
        Severity scores computed in the first 24 hours of patient admission.
    """
    sirs = """SELECT sirs.subject_id, sirs.hadm_id, sirs.sirs FROM `physionet-data.mimiciii_derived.sirs` sirs"""
    sirs = client.query(sirs).result().to_dataframe().rename(str.lower, axis='columns')
    sapsii = ("SELECT sapsii.sapsii, sapsii.hadm_id, sapsii.subject_id "
              "FROM `physionet-data.mimiciii_derived.sapsii` sapsii ")
    sapsii = client.query(sapsii).result().to_dataframe().rename(str.lower, axis='columns')
    apsiii = """SELECT apsiii.subject_id, apsiii.hadm_id, apsiii.apsiii
     FROM `physionet-data.mimiciii_derived.apsiii` apsiii"""
    apsiii = client.query(apsiii).result().to_dataframe().rename(str.lower, axis='columns')
    sofa = ("SELECT sofa.sofa, sofa.hadm_id, sofa.subject_id "
            "FROM `physionet-data.mimiciii_derived.sofa` sofa ")
    sofa = client.query(sofa).result().to_dataframe().rename(str.lower, axis='columns')
    lods = """SELECT lods.subject_id, lods.hadm_id, lods.lods 
    FROM `physionet-data.mimiciii_derived.lods` lods"""
    lods = client.query(lods).result().to_dataframe().rename(str.lower, axis='columns')
    oasis = ("SELECT oasis.oasis, oasis.hadm_id, oasis.subject_id"
             " FROM `physionet-data.mimiciii_derived.oasis` oasis ")
    oasis = client.query(oasis).result().to_dataframe().rename(str.lower, axis='columns')

    severity_score = sirs.merge(sapsii, how='inner', on=['hadm_id', 'subject_id'])
    severity_score = severity_score.merge(apsiii, how='inner', on=['hadm_id', 'subject_id'])
    severity_score = severity_score.merge(sofa, how='inner', on=['hadm_id', 'subject_id'])
    severity_score = severity_score.merge(lods, how='inner', on=['hadm_id', 'subject_id'])
    severity_score = severity_score.merge(oasis, how='inner', on=['hadm_id', 'subject_id'])
    severity_score = severity_score.set_index(['hadm_id', 'subject_id'])
    severity_score = severity_score[~severity_score.index.duplicated(keep='first')]
    severity_score = severity_score.reset_index()
    return severity_score


def load_meld(client):
    """
        The MELD score, often used to assess health of liver transplant candidates.
        Relates to organ failure.
        Compute based on first-day of admission data.
        MELD used to predict 3-month mortality due to liver disease.
        MELD scores range from 6 to 40; the higher the score, the higher the 3-month mortality related to liver disease.
    """
    meld = """SELECT meld.subject_id, meld.hadm_id, meld.meld FROM `physionet-data.mimiciii_derived.meld` meld"""
    meld = client.query(meld).result().to_dataframe().rename(str.lower, axis='columns')
    return meld


def load_aki(client, map_icustay_id):
    """
    A variation of the code from MIMIC code repository:
    https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/organfailure/kdigo_stages_48hr.sql
    Check if the patient had AKI during the first 42 hours of this ICU stay.
    Return a categorical variable.

    :param client:          BigQuery client
    :param map_icustay_id:  (pd.DataFrame) index by icustay_id columns are 'subject_id', 'hadm_id'.
                            To map between the icu stay id to the subject and hadm ids.
    :return:
    """
    KDIGO_Stages_query = """-- This query checks if the patient had AKI during the first 42 hours of their ICU
    -- stay according to the KDIGO guideline.
    -- https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf

    -- get the worst staging of creatinine in the first 42 hours
    WITH cr_aki AS
    (
      SELECT
        k.icustay_id
        , k.charttime
        , k.creat
        , k.aki_stage_creat
        , ROW_NUMBER() OVER (PARTITION BY k.icustay_id ORDER BY k.aki_stage_creat DESC, k.creat DESC) AS rn
      FROM `physionet-data.mimiciii_clinical.icustays` ie
      INNER JOIN `physionet-data.mimiciii_derived.kdigo_stages` k
        ON ie.icustay_id = k.icustay_id
      WHERE DATETIME_DIFF(k.charttime, ie.intime, HOUR) > -6
      AND DATETIME_DIFF(k.charttime, ie.intime, HOUR) <= 42
      AND k.aki_stage_creat IS NOT NULL
    )
    -- get the worst staging of urine output in the first 42 hours
    , uo_aki AS
    (
      SELECT
        k.icustay_id
        , k.charttime
        , k.uo_rt_6hr, k.uo_rt_12hr, k.uo_rt_24hr
        , k.aki_stage_uo
        , ROW_NUMBER() OVER 
        (
          PARTITION BY k.icustay_id
          ORDER BY k.aki_stage_uo DESC, k.uo_rt_24hr DESC, k.uo_rt_12hr DESC, k.uo_rt_6hr DESC
        ) AS rn
      FROM `physionet-data.mimiciii_clinical.icustays` ie
      INNER JOIN `physionet-data.mimiciii_derived.kdigo_stages` k
        ON ie.icustay_id = k.icustay_id
      WHERE DATETIME_DIFF(k.charttime, ie.intime, HOUR) > -6
      AND DATETIME_DIFF(k.charttime, ie.intime, HOUR) <= 42
      AND k.aki_stage_uo IS NOT NULL
    )
    -- final table is aki_stage, include worst cr/uo for convenience
    select
        ie.icustay_id
      , cr.charttime as charttime_creat
      , cr.creat
      , cr.aki_stage_creat
      , uo.charttime as charttime_uo
      , uo.uo_rt_6hr
      , uo.uo_rt_12hr
      , uo.uo_rt_24hr
      , uo.aki_stage_uo

      -- Classify AKI using both creatinine/urine output criteria
      , GREATEST(
          COALESCE(cr.aki_stage_creat, 0),
          COALESCE(uo.aki_stage_uo, 0)
        ) AS aki_stage_42hr
      , CASE WHEN cr.aki_stage_creat > 0 OR uo.aki_stage_uo > 0 THEN 1 ELSE 0 END AS aki_42hr

    FROM `physionet-data.mimiciii_clinical.icustays` ie
    LEFT JOIN cr_aki cr
      ON ie.icustay_id = cr.icustay_id
      AND cr.rn = 1
    LEFT JOIN uo_aki uo
      ON ie.icustay_id = uo.icustay_id
      AND uo.rn = 1
    order by ie.icustay_id;"""

    KDIGO_Stages = client.query(KDIGO_Stages_query).result().to_dataframe().rename(str.lower, axis='columns')
    aki_42hr = KDIGO_Stages.set_index('icustay_id').join(map_icustay_id,
                                                         how='inner').set_index(['subject_id', 'hadm_id'])['aki_42hr']
    aki_42hr = aki_42hr[~aki_42hr.index.duplicated(keep='first')]
    return aki_42hr


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_model(pipe, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pipe, f)
