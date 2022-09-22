from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, jaccard_score, \
    roc_curve, auc, cohen_kappa_score
import matplotlib.pyplot as plt


def calculate_test_metrics(y_true, y_pred):
    """
    Calculates a number of evaluation metrics.

    """

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    matt_cor_coef = matthews_corrcoef(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return accuracy, precision, recall, f1, matt_cor_coef, jaccard, kappa


def plot_roc_curve(y_true, y_pred, run_id):
    """
    Display a Receiver Operating Characteristic curve.

    """

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc = auc(fpr,tpr)

    fig, ax = plt.subplots(1, 1)
    ax.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.savefig(run_id + '_roc.png')