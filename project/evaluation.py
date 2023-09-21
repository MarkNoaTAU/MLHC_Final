from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve, roc_curve, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, average_precision_score, \
    confusion_matrix, ConfusionMatrixDisplay
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibrationDisplay

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap

NUM_RESAMPLE = 20
SAMPLE_PERCENTAGE = 0.2


def compute_pr80(proba_true_class, y_true):
    """
    Compute model recall at precision of 80%.
    Important in evaluating of health-care prediction models, as it take into account that we can not tolerance low
    precision - to avoid alarm fatigue.
    :param proba_true_class:  (pd.Series) Probability score for the true class.
    :param y_true:            (pd.Series) Target
    :return:  Model recall at precision of 80%.
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(probas_pred=proba_true_class, y_true=y_true)
    return recall_arr[precision_arr >= 0.8].max()


def print_eval_stats(pipe, X_to_eval, y_to_eval, subset_str='test'):
    """
    Calculate and print different evaluation matrices: acc, balance acc, f1 score, PR80, AP.
    As well as print confusion matrix of the given data.
    :param pipe:        (sklearn.pipeline.Pipeline) Model
    :param X_to_eval:   (pd.DataFrame) Features
    :param y_to_eval:   (pd.Series) Target
    :param subset_str:  (str) description of the data - train / val / test
    """
    pred_to_eval = pipe.predict(X_to_eval)
    pred_proba = pipe.predict_proba(X_to_eval)
    proba_true_class = pred_proba[:, 1]

    pr80 = compute_pr80(proba_true_class, y_to_eval)
    acc = accuracy_score(y_pred=pred_to_eval, y_true=y_to_eval)
    f1_s = f1_score(y_pred=pred_to_eval, y_true=y_to_eval)
    balanced_acc = balanced_accuracy_score(y_pred=pred_to_eval, y_true=y_to_eval)
    ap = average_precision_score(y_score=proba_true_class, y_true=y_to_eval)

    print(f"Evaluation (on {subset_str}): acc - {round(acc, 2)}, balanced_acc - {round(balanced_acc, 2)},"
          f" \n f1 score - {round(f1_s, 2)}, PR80 - {round(pr80, 2)}, ap- {round(ap, 2)}")
    cm = confusion_matrix(y_to_eval, pred_to_eval)
    ConfusionMatrixDisplay(cm).plot()
    return pd.DataFrame([acc, balanced_acc, f1_s, pr80, ap], columns=[subset_str],
                        index=['acc', 'balanced_acc', 'f1_score', 'PR80', 'ap'])


def plot_roc_pr_curves(y_, y_pred_proba_, plot_title):
    """
    Plot ROC & PR curves with bootstrap confidence evaluation.

    Took inspiration from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    :param y_:            (pd.Series) Target
    :param y_pred_proba_: (pd.Series) Model prediction probability scores for the target.
    :param plot_title:    (str) plot title
    """
    n_samples = int(y_.shape[0] * SAMPLE_PERCENTAGE)

    def plot_std(y_bs, x, ax):
        mean_y = np.mean(y_bs, axis=0)
        std_y = np.std(y_bs, axis=0)

        y_upper = np.minimum(mean_y + std_y, 1)
        y_lower = np.maximum(mean_y - std_y, 0)

        ax.fill_between(x, y_lower, y_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.", )

    proba_true_class = y_pred_proba_[:, 1]
    # plot the full set:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    RocCurveDisplay.from_predictions(y_true=y_, y_pred=proba_true_class, ax=ax1)
    PrecisionRecallDisplay.from_predictions(y_true=y_, y_pred=proba_true_class, ax=ax2)

    # create the standard deviation statistics:
    tprs = []
    precisions = []
    fpr_mean = np.linspace(0, 1, 1000)
    recall_mean = np.linspace(0, 1, 1000)
    for i in range(NUM_RESAMPLE):
        pred_s, true_s = resample(proba_true_class, y_, replace=True, n_samples=n_samples)
        fpr, tpr, _ = roc_curve(true_s, pred_s)
        precision, recall, _ = precision_recall_curve(true_s, pred_s)
        precision, recall = precision[::-1], recall[::-1]

        interp_tpr = np.interp(fpr_mean, xp=fpr, fp=tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        interp_precision = np.interp(recall_mean, xp=recall, fp=precision)
        precisions.append(interp_precision)

    plot_std(y_bs=tprs, x=fpr_mean, ax=ax1)
    plot_std(y_bs=precisions, x=recall_mean, ax=ax2)

    ax1.set(xlabel="FPR", ylabel="TPR", title=f"ROC with bootstrap confidence evaluation", )
    ax2.set(xlabel="Recall", ylabel="Precision", title=f"PR with bootstrap confidence evaluation", )
    fig.suptitle(plot_title)
    plt.show()


def plot_model_feature_importance(model, ft):
    """
    Plot bar plot including std of the feature importance calculated in Random Forest implementation.

    :param model: (Sklearn Model)
    :param ft:    (pd.DataFrame) model's feature
    """
    if isinstance(model, BalancedRandomForestClassifier) or isinstance(model, RandomForestClassifier):
        feature_names = ft.columns
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

        forest_importances = pd.DataFrame({'importances': importances, 'std': std}, index=feature_names).sort_values(
            by='importances', ascending=False)

        fig, ax = plt.subplots(figsize=(30, 20))
        forest_importances['importances'].plot.bar(yerr=forest_importances['std'], ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()


def plot_shap_feature_importance_values(model, ft_test):
    """
    Plot the SHAP values of the True class.
    Note, that currently, SHAP does not support imblearn.ensemble._bagging.BalancedBaggingClassifier,
    only Random-Forest like trees.
    :param model:   (Sklearn model)
    :param ft_test: (pd.DataFrame) features for the test set.
    :return: shap_values
    """
    if isinstance(model, BalancedRandomForestClassifier) or isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(ft_test)
        shap.summary_plot(shap_values[1], ft_test, plot_type="bar", max_display=20, class_names=None)
        return shap_values


def plot_calibration(pred_proba, y_test):
    """

    :param pred_proba: (numpy array) containing the prediction probability for the False and True classes.
    :param y_test:     (pd.Series) the target
    :return:
    """
    pred_proba = pd.DataFrame(pred_proba, columns=['False', 'True'])
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # fig, ax = plt.subplots()
    pred_proba = pd.DataFrame(pred_proba, columns=['False', 'True'])

    CalibrationDisplay.from_predictions(y_test, pred_proba['True'], n_bins=10, ax=ax1)
    # CalibrationDisplay.from_predictions(y_test, pred_proba['True'], n_bins=10, zorder=1, ax=ax)

    pred_proba.plot.hist(bins=100, ax=ax2, title="Prediction Probability Histogram")
    # pred_proba.plot.hist(bins=100, zorder=2, ax=ax)
    ax1.set(title="Reliability diagram")
    plt.show()
