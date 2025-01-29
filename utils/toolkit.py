import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy, for initial classes
    idxes = np.where(
        np.logical_and(y_true >= 0, y_true < init_cls)
    )[0]
    label = "{}-{}".format(
        str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
    )
    all_acc[label] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )
    # for incremental classes
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc

from sklearn.metrics import f1_score

def f1_score_custom(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_f1 = {}

    # Total F1 score
    all_f1["total"] = np.around(f1_score(y_true, y_pred, average='weighted'), decimals=2)

    # Grouped F1 score for initial classes
    idxes = np.where(np.logical_and(y_true >= 0, y_true < init_cls))[0]
    label = "{}-{}".format(str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0"))
    all_f1[label] = np.around(f1_score(y_true[idxes], y_pred[idxes], average='weighted'), decimals=2)

    # For incremental classes
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_f1[label] = np.around(f1_score(y_true[idxes], y_pred[idxes], average='weighted'), decimals=2)

    # Old F1 score
    idxes = np.where(y_true < nb_old)[0]
    all_f1["old"] = 0 if len(idxes) == 0 else np.around(f1_score(y_true[idxes], y_pred[idxes], average='weighted'), decimals=2)

    # New F1 score
    idxes = np.where(y_true >= nb_old)[0]
    all_f1["new"] = np.around(f1_score(y_true[idxes], y_pred[idxes], average='weighted'), decimals=2)

    return all_f1


from sklearn.metrics import matthews_corrcoef
import numpy as np

def mcc_score_custom(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_mcc = {}

    # Total MCC score
    all_mcc["total"] = np.around(matthews_corrcoef(y_true, y_pred), decimals=2)

    # Grouped MCC score for initial classes
    idxes = np.where(np.logical_and(y_true >= 0, y_true < init_cls))[0]
    label = "{}-{}".format(str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0"))
    all_mcc[label] = np.around(matthews_corrcoef(y_true[idxes], y_pred[idxes]), decimals=2)

    # For incremental classes
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_mcc[label] = np.around(matthews_corrcoef(y_true[idxes], y_pred[idxes]), decimals=2)

    # Old MCC score
    idxes = np.where(y_true < nb_old)[0]
    all_mcc["old"] = 0 if len(idxes) == 0 else np.around(matthews_corrcoef(y_true[idxes], y_pred[idxes]), decimals=2)

    # New MCC score
    idxes = np.where(y_true >= nb_old)[0]
    all_mcc["new"] = np.around(matthews_corrcoef(y_true[idxes], y_pred[idxes]), decimals=2)

    return all_mcc

from sklearn.metrics import cohen_kappa_score

def kappa_score_custom(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_kappa = {}

    # Total Kappa score
    all_kappa["total"] = np.around(cohen_kappa_score(y_true, y_pred), decimals=2)

    # Grouped Kappa score for initial classes
    idxes = np.where(np.logical_and(y_true >= 0, y_true < init_cls))[0]
    label = "{}-{}".format(str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0"))
    all_kappa[label] = np.around(cohen_kappa_score(y_true[idxes], y_pred[idxes]), decimals=2)

    # For incremental classes
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_kappa[label] = np.around(cohen_kappa_score(y_true[idxes], y_pred[idxes]), decimals=2)

    # Old Kappa score
    idxes = np.where(y_true < nb_old)[0]
    all_kappa["old"] = 0 if len(idxes) == 0 else np.around(cohen_kappa_score(y_true[idxes], y_pred[idxes]), decimals=2)

    # New Kappa score
    idxes = np.where(y_true >= nb_old)[0]
    all_kappa["new"] = np.around(cohen_kappa_score(y_true[idxes], y_pred[idxes]), decimals=2)

    return all_kappa

from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix
import numpy as np

def balanced_accuracy_custom(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_balanced_acc = {}

    # Total Balanced Accuracy
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fn + fp)
    
    # Handle potential division by zero
    balanced_accuracy_total = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
    balanced_accuracy_total = np.nan_to_num(balanced_accuracy_total, nan=0)  # Replace NaN with 0
    all_balanced_acc["total"] = np.around(np.mean(balanced_accuracy_total), decimals=2)

    # Grouped Balanced Accuracy for initial classes
    idxes = np.where(np.logical_and(y_true >= 0, y_true < init_cls))[0]
    if len(idxes) > 0:
        cm_init = confusion_matrix(y_true[idxes], y_pred[idxes])
        tp_init = np.diag(cm_init)
        fn_init = cm_init.sum(axis=1) - tp_init
        fp_init = cm_init.sum(axis=0) - tp_init
        tn_init = cm_init.sum() - (tp_init + fn_init + fp_init)

        # Handle potential division by zero
        balanced_accuracy_init = 0.5 * ((tp_init / (tp_init + fn_init)) + (tn_init / (tn_init + fp_init)))
        balanced_accuracy_init = np.nan_to_num(balanced_accuracy_init, nan=0)  # Replace NaN with 0
        label_init = "{}-{}".format(str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0"))
        all_balanced_acc[label_init] = np.around(np.mean(balanced_accuracy_init), decimals=2)
    else:
        all_balanced_acc["init"] = 0  # If no data for initial classes, set to 0

    # For incremental classes
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        if len(idxes) > 0:
            cm_increment = confusion_matrix(y_true[idxes], y_pred[idxes])
            tp_increment = np.diag(cm_increment)
            fn_increment = cm_increment.sum(axis=1) - tp_increment
            fp_increment = cm_increment.sum(axis=0) - tp_increment
            tn_increment = cm_increment.sum() - (tp_increment + fn_increment + fp_increment)

            # Handle potential division by zero
            balanced_accuracy_increment = 0.5 * ((tp_increment / (tp_increment + fn_increment)) + (tn_increment / (tn_increment + fp_increment)))
            balanced_accuracy_increment = np.nan_to_num(balanced_accuracy_increment, nan=0)  # Replace NaN with 0
            label_increment = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
            all_balanced_acc[label_increment] = np.around(np.mean(balanced_accuracy_increment), decimals=2)
        else:
            all_balanced_acc[f"{class_id}-{class_id + increment - 1}"] = 0  # If no data for this increment, set to 0

    # Old Balanced Accuracy
    idxes = np.where(y_true < nb_old)[0]
    if len(idxes) > 0:
        cm_old = confusion_matrix(y_true[idxes], y_pred[idxes])
        tp_old = np.diag(cm_old)
        fn_old = cm_old.sum(axis=1) - tp_old
        fp_old = cm_old.sum(axis=0) - tp_old
        tn_old = cm_old.sum() - (tp_old + fn_old + fp_old)

        # Handle potential division by zero
        balanced_accuracy_old = 0.5 * ((tp_old / (tp_old + fn_old)) + (tn_old / (tn_old + fp_old)))
        balanced_accuracy_old = np.nan_to_num(balanced_accuracy_old, nan=0)  # Replace NaN with 0
        all_balanced_acc["old"] = np.around(np.mean(balanced_accuracy_old), decimals=2)
    else:
        all_balanced_acc["old"] = 0  # No data for old classes, set to 0

    # New Balanced Accuracy
    idxes = np.where(y_true >= nb_old)[0]
    if len(idxes) > 0:
        cm_new = confusion_matrix(y_true[idxes], y_pred[idxes])
        tp_new = np.diag(cm_new)
        fn_new = cm_new.sum(axis=1) - tp_new
        fp_new = cm_new.sum(axis=0) - tp_new
        tn_new = cm_new.sum() - (tp_new + fn_new + fp_new)

        # Handle potential division by zero
        balanced_accuracy_new = 0.5 * ((tp_new / (tp_new + fn_new)) + (tn_new / (tn_new + fp_new)))
        balanced_accuracy_new = np.nan_to_num(balanced_accuracy_new, nan=0)  # Replace NaN with 0
        all_balanced_acc["new"] = np.around(np.mean(balanced_accuracy_new), decimals=2)
    else:
        all_balanced_acc["new"] = 0  # No data for new classes, set to 0

    return all_balanced_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
