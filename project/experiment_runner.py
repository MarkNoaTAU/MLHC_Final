import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier

from project.utilities import create_pipeline
from project.data_loaders import MimicExtractDataLoader
from project.evaluation import plot_shap_feature_importance_values
from project.configs import ModelType


MAP_FNAME = {ModelType.RANDOM_FOREST: {'model__max_depth': 'max_depth',
                                       'model__min_samples_split': 'min_samples_split',
                                       'model__max_features': 'max_features',
                                       'model__n_estimators': 'n_estimators'},
             ModelType.GRADIENT_BOOSTING: {'model__estimator__max_depth': 'max_depth',
                                            'model__estimator__max_iter': 'max_iter',
                                            'model__estimator__max_leaf_nodes': 'max_leaf_nodes',
                                            'model__estimator__learning_rate': 'learning_rate'}
             }

N_SPLITS = 5
DEF_PARAMS = {ModelType.RANDOM_FOREST: dict(model__max_depth=[6, 8, 10, 12],
                                            model__min_samples_split=[2, 4],
                                            model__max_features=[7, 15],
                                            model__n_estimators=[100]),
              ModelType.GRADIENT_BOOSTING: dict(
                  model__estimator__max_iter=[100],
                  model__estimator__max_leaf_nodes=[5, 10, 30, 50],
                  model__estimator__learning_rate=[0.01, 0.1, 1])
              }


def hyper_parameter_tunner(X_train, y_train, params=None, random_state=0, exp_model_type=ModelType.GRADIENT_BOOSTING,
                           to_tune=True, scoring='f1'):
    """

    :param X_train:         (pd.Series) features
    :param y_train:         (pd.Series) target
    :param params:          (Dict or None) if None search on default grid search, else search according to the provided
                            parameter grid.
    :param random_state:    (int) random state
    :param exp_model_type:  (ModelType) the model type
    :param to_tune:         (bool) if False return empty dict else, search for the best hyperparameters.
    :param scoring:
    :return:                (Dict) the best hyperparameter for the model according to the scoring,
                            given grid search on params, using cv split.

    """
    if not to_tune:
        return dict()
    categrical_features = X_train.columns.intersection(MimicExtractDataLoader.CATEGORICAL_FEATURES)
    intervention_features = X_train.columns.intersection(MimicExtractDataLoader.INTERVENTION_FEATURES)
    numerical_features = X_train.columns.drop(categrical_features).drop(intervention_features)

    if exp_model_type == ModelType.RANDOM_FOREST:
        model = BalancedRandomForestClassifier(random_state=random_state)
    elif exp_model_type == ModelType.GRADIENT_BOOSTING:
        model = BalancedBaggingClassifier(
            estimator=HistGradientBoostingClassifier(random_state=random_state))
    else:
        raise ValueError("Model type provided is not supported")

    pipe = create_pipeline(model, numerical_features)
    cv = StratifiedKFold(n_splits=N_SPLITS).split(X_train, y_train)
    params = DEF_PARAMS[exp_model_type] if params is None else params
    cv_model = GridSearchCV(pipe, param_grid=params, cv=cv, scoring=scoring, verbose=4)
    cv_model.fit(X_train, y_train)
    print(cv_model.best_params_)

    map_fnames = MAP_FNAME[exp_model_type]
    best_hyper_parameters = pd.Series(cv_model.best_params_).rename(map_fnames).to_dict()
    return best_hyper_parameters


def feature_selector(model, X_test, k=75):
    """
    Select and plot SHAP importance values.
    Note, will always return Age and Gender as part of the final list - for imputation based on their grouping.
    :param model:   (sklearn model)
    :param X_test:  (pd.DataFrame) test features
    :param k:       (int) Number of features
    :return: list of the top k features selected according to SHAP values
    """

    shap_values = plot_shap_feature_importance_values(model, X_test)
    feature_importance = pd.DataFrame(shap_values[1], columns=X_test.columns).abs().mean().sort_values(ascending=False)
    k_top_features = feature_importance[: k].index.to_list()
    if 'gender' not in k_top_features:
        print("gender not in top K features, initally")
        k_top_features.append('gender')
    if 'age' not in k_top_features:
        print("age not in top K features, initally")
        k_top_features.append('age')
        print(f"Top K (k= {k}) features: {k_top_features}")
    return k_top_features
