import io
import itertools
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def FPR_AT_95_TPR(TPR, FPR):
    """  Measures the false positive rate when the true positive rate is equal to 95%
        TPR -> list: True Positive Rate
        FPR -> list: False Positive Rate
    return:
        fpr_at_95_tpr -> float: the fpr_at_95_tpr
    """
    for i in range(len(TPR)):
        if 0.9505 >= TPR[i] >= 0.9495:
            return FPR[i]
    return 0


def ece(results, precision=15):
    """ Expected Calibration Error (ECE)
        results   -> tensor: (uncertainty, prediction, labels), the tensor has to be sorted by uncertainty
        precision -> int: numbers of bins
    return:
        ece_score -> float: the ece score
        tab_conf  -> list: the list for each bin of the ocnfidence score
        tab_acc   -> list: the list for each bin of the accuracy score
    """
    res = results[:, 0] / torch.max(results[:, 0])
    bin_boundaries = torch.linspace(0, 1, precision + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = 1 - res
    accuracies = results[:, -1].eq(results[:, -2])
    tab_acc, tab_conf = [], []
    ece_score = torch.zeros(1, device=results.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece_score += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            tab_acc.append(accuracy_in_bin)
            tab_conf.append(avg_confidence_in_bin)
    return ece_score.item(), tab_conf, tab_acc


def ace(results, precision):
    """ Adaptive Calibration Error (ACE)
        results -> tensor: (uncertainty, prediction, labels), the tensor has to be sorted by uncertainty
        precision -> int: numbers of bins
    return:
        ace_score -> float: the ace score
    """

    results[:, 0] /= torch.max(results[:, 0])                                    # Standardize the input between 0 and 1
    ace_score = torch.zeros(1, device=results.device)
    nb_classes = torch.unique(results[:, -1])                                    # All possible classes
    for k in nb_classes:
        res_k = results[results[:, -1] == k]                                     # Select only the classe k
        acc_k = res_k[:, -1].eq(res_k[:, -2])
        conf_k = 1 - res_k[:, 0]                                                 # Convert uncertainty to confidence
        bin_boundaries = np.linspace(0, len(res_k), precision, dtype=int)
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            accuracy_in_bin = acc_k[bin_lower:bin_upper].float().mean()
            avg_confidence_in_bin = conf_k[bin_lower:bin_upper].float().mean()
            ace_score += torch.abs(accuracy_in_bin - avg_confidence_in_bin)
    ace_score /= (len(nb_classes) * precision)
    return ace_score.item()


def compute_score(tab, args):
    """ Compute different kind of metrics to evaluate OOD detection
        tab  -> tensor: (uncertainty, prediction, labels), the prediction with the confidence score
        args ->  Argparse: global arguments
    return:
        acc          -> the accuracy
        aupr_success -> area under the precison-recall curve where positive class are correct predictions
        aupr_error   -> area under the precison-recall curve where positive class are errors
        auroc        -> area under the roc curve
        fpr_at_95tpr -> false positive rate when the true positive rate is equal to 95%
        ace_score    -> Adaptive Calibration Error
        ece_score    -> Expected Calibration Error
    """

    tab = tab[tab[:, 0].argsort()]
    tab = tab.to(args.device)

    TPR, FPR = [], []
    list_P_sucess, list_R_sucess = [], []
    list_P_error, list_R_error = [], []
    for ind, i in enumerate(torch.linspace(0, len(tab), 10_000)):
        if ind == 0:
            continue
        i = int(i)
        TP = torch.where(tab[:i, -2] == tab[:i, -1], args.one, args.zero).sum().cpu().item()
        FP = torch.where(tab[:i, -2] != tab[:i, -1], args.one, args.zero).sum().cpu().item()
        FN = torch.where(tab[i:, -2] == tab[i:, -1], args.one, args.zero).sum().cpu().item()
        TN = torch.where(tab[i:, -2] != tab[i:, -1], args.one, args.zero).sum().cpu().item()

        TPR.append(TP / (TP + FN + 1e-11))
        FPR.append(FP / (FP + TN + 1e-11))
        list_P_sucess.append(TP / (TP + FP + 1e-11))
        list_R_sucess.append(TP / (TP + FN + 1e-11))
        list_P_error.insert(0, TN / (TN + FN + 1e-11))
        list_R_error.insert(0, TN / (TN + FP + 1e-11))

    #acc = tab[:, -2].eq(tab[:, -1]).cpu().numpy().mean()
    aupr_success = np.trapz(list_P_sucess, list_R_sucess)
    aupr_error = np.trapz(list_P_error, list_R_error)
    auroc = np.trapz(TPR, FPR)
    fpr_at_95tpr = FPR_AT_95_TPR(TPR, FPR)
    ace_score = ace(tab.cpu(), 15)

    return [aupr_success, aupr_error, auroc, fpr_at_95tpr, ace_score]


def print_result(name, split, res, epoch, args):
    """ Print and return the evaluation of the prediction
        name      -> str: name of the method
        split     -> str: either Train, Val or Test
        res       -> tensor: (uncertainty, prediction, labels), the prediction with the confidence score
        epoch     -> int: current epoch
        args      -> Argparse: global arguments
    return:
        results -> dict: the all the metrics computed
    """

    aupr_success, aupr_error, auroc, fpr_at_95tpr, ace = compute_score(res, args)

    s = "\r" + split + " " + name
    s += ': FPR@95 {:.1f}, AuPR sucess: {:.1f}, AuPR error: {:.1f}, AuROC: {:.1f},  ACE {:.3f}'
    print(s.format(fpr_at_95tpr * 100, aupr_success * 100, aupr_error * 100, auroc * 100, ace))

    results = {"auroc": auroc * 100, "aupr": aupr_success * 100, "fpr_at_95tpr": fpr_at_95tpr * 100, "ace": ace, "ece": ece}
    return results

class ConfusionMatrix(object):
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, 'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(),  'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), 'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), 'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self, norm=False):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized or norm:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

    def plot_confusion_matrix(self, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(16, 16))
        plt.imshow(self.value(True), interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        cm = np.around(self.conf.astype('float') / (self.conf.sum(axis=1)[:, np.newaxis]+1e-11), decimals=2)

        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = transforms.ToTensor()(Image.open(buf))
        return image


class IoU(object):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()

        if ignore_index is None or ignore_index < 0:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
            num_classes += len(self.ignore_index)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

        self.conf_metric = ConfusionMatrix(num_classes, normalized)

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.
        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.
        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.
        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            conf_matrix = self.conf_metric.value()
            if self.ignore_index is not None:
                for _ in self.ignore_index:
                    conf_matrix[:, self.ignore_index] = 0
                    conf_matrix[self.ignore_index, :] = 0
            true_positive = np.diag(conf_matrix)
            false_positive = np.sum(conf_matrix, 0) - true_positive
            false_negative = np.sum(conf_matrix, 1) - true_positive
            iou = true_positive / (true_positive + false_positive + false_negative)

        class_avg = self.conf_metric.value(True).diagonal().mean()

        return iou, np.nanmean(iou), class_avg

    def plot_confusion_matrix(self, c):
        return self.conf_metric.plot_confusion_matrix(c)