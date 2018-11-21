import sys
import os
import numpy as np
import tensorflow as tf
import six
import subprocess, re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import logging
import contextlib
import math

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import device_setter
from tqdm import tqdm

import sklearn.metrics as metrics


# https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def get_onehot(label):
    """
    get one hot encoded label matrix
    Args:
        label: original label
    Returns:
        One hot encoded label matrix
    """
    # Make 0 initialized numpy array with shape of [label.shape[0], 10]\n",
    one_hot = np.zeros((label.shape[0], len(np.unique(label))), dtype=np.int)
    # Fill up the array according to the input label\n",
    one_hot[np.arange(label.shape[0]), label.astype(int)] = 1
    return one_hot


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def pprint_dict(dictionary, title):
    print('\n[*] {}.... '.format(title) + '=' * 30)
    for key in sorted(list(dictionary.keys())):
        if 'dir' in key:
            print('{:>20} : {:<30}'.format(key, 'None' if dictionary[key] is None else os.path.realpath(dictionary[key])))
        else:
            print('{:>20} : {:<30}'.format(key, 'None' if dictionary[key] is None else dictionary[key]))
    print('=' * 53)
    print('\n')


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


class Logger(object):
    def __init__(self, terminal, logger, log_level=logging.INFO):
        self.terminal = terminal
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
        self.terminal.write(buf)

    def flush(self):
        pass


def average_gradient(tower_grads):
    average_grads = []
    for grads_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grads_and_vars:
            expanded_g = tf.expand_dims(g, 0)

            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grads_and_vars[0][1]
        grads_and_vars = (grad, v)
        average_grads.append(grads_and_vars)

    return average_grads


def get_one_hot(label, num_class):
    one_hot = np.zeros((label.shape[0], num_class))
    one_hot[np.arange(label.shape[0]), label.astype(int)] = 1
    return one_hot


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops is None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()
    return _local_device_chooser


def run_command(cmd, decoding):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode(decoding)


def get_available_gpus(num_gpus=1, memory_fraction=100):
    """Returns GPU with the least allocated memory"""
    def list_available_gpus():
        """Returns list of available GPU ids."""
        output = run_command("nvidia-smi -L", 'ascii')
        # lines of the form GPU 0: TITAN X
        gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
        result = []
        for line in output.strip().split("\n"):
            m = gpu_regex.match(line)
            assert m, "Couldnt parse " + line
            result.append(int(m.group("gpu_id")))
        return result

    def gpu_memory_map():
        """Returns map of GPU id to memory allocated on that GPU."""

        output = run_command("nvidia-smi", 'ascii')
        total_output = output[output.find("Memory-Usage"):]
        row = total_output.split("\n")[3]
        total_memory = int(re.compile(r"\s/\s(?P<total_memory>\d+)MiB").search(row).group('total_memory'))

        gpu_output = output[output.find("GPU Memory"):]
        # lines of the form
        # |    0      8734    C   python                                       11705MiB |
        memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
        rows = gpu_output.split("\n")
        result = {gpu_id: 0 for gpu_id in list_available_gpus()}
        for row in gpu_output.split("\n"):
            m = memory_regex.search(row)
            if not m:
                continue
            gpu_id = int(m.group("gpu_id"))
            gpu_memory = int(m.group("gpu_memory"))
            result[gpu_id] += gpu_memory
        return result, total_memory

    result, total_memory = gpu_memory_map()
    bound = math.floor(total_memory * memory_fraction / 100.)

    memory_gpu_map = np.array([(memory, gpu_id) for (gpu_id, memory) in result.items()])

    if num_gpus > len(memory_gpu_map):
        raise ValueError('Total number of gpus in this machine is {} but {} is requested'.format(
            len(memory_gpu_map), num_gpus))

    memory_gpu_map[:, 0] = total_memory - memory_gpu_map[:, 0]
    memory_gpu_map = np.stack(sorted(memory_gpu_map.tolist()))
    gpu_memory = memory_gpu_map[:, 0]
    gpu_idx = memory_gpu_map[:, 1]

    selected_gpus = []
    for i in range(len(gpu_idx)):
        if gpu_memory[i] >= bound:
            selected_gpus.append(gpu_idx[i])

    if len(selected_gpus) < num_gpus:
        raise ValueError('{} gpus are not available at this point. {} gpus are available.'.format(
            num_gpus, len(selected_gpus)))
    selected_gpus = np.array(selected_gpus[0:num_gpus])

    return re.sub('[\[\]]', '', np.array2string(selected_gpus, precision=0, separator=','))


def get_confusion_matrix(label, prediction, decision_bound=None, verbose=False):
    if decision_bound:
        prediction = (prediction >= decision_bound).astype(int)
    else:
        prediction = prediction.round()
    cnf_matrix = metrics.confusion_matrix(label, prediction.round())

    FP = cnf_matrix[0, 1]
    FN = cnf_matrix[1, 0]
    TP = cnf_matrix[1, 1]
    TN = cnf_matrix[0, 0]

    if verbose:
        from tabulate import tabulate
        print(tabulate([['{:.3f}'.format(decision_bound), 'condition\npositive', 'condition\nnegative'],
                        ['predicted\npositive', '{}\n(TP)'.format(TP), '{}\n(FP)'.format(FP)],
                        ['predicted\nnegative', '{}\n(FN)'.format(FN), '{}\n(TN)'.format(TN)]],
                       tablefmt='fancy_grid'))

    return FP, FN, TP, TN


def get_metrics(FP, FN, TP, TN, verbose=False):
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    if verbose:
        print('True Positive rate(recall, sensitivity): {:.4f}'.format(TPR))
        print('True negative rate(specificity): {:.4f}'.format(TNR))
        print('Positive predictive value(precision): {:.4f}'.format(PPV))
        print('Negative predictive value: {:.4f}'.format(NPV))
        print('False posivite rate: {:.4f}'.format(FPR))
        print('False negative rate: {:.4f}'.format(FNR))
        print('False discovery rate: {:.4f}'.format(FDR))
        print('\n')

    return TPR, TNR, PPV, NPV, FPR, FNR, FDR


def get_plot_args(lw=None, alpha=None, color=None):
    return {'lw': lw,
            'alpha': alpha,
            'color': color}


def plot_calibration_analysis(fignum, label, prediction, func=None, bin=10, **kwargs):
    lw = kwargs['lw']
    alpha = kwargs['alpha']
    color = kwargs['color']

    bin_start_list = []
    bin_end_list = []
    bin_size = 0
    if func is None:
        bin_size = 1. / bin
        bin_start_list = np.arange(0, 1, bin_size)
        bin_end_list = bin_start_list + bin_size
    #  elif func == 'linear':

    x = ['[{:.2f}, {:.2f})'.format(bin_start_list[i], bin_end_list[i]) for i in range(bin)]
    #  x = [(bin_start_list[i] + bin_end_list[i]) / 2 for i in range(bin)]
    y = []
    ps = []
    ns = []
    for i in range(bin):
        bin_label = label[np.squeeze(np.array(prediction >= bin_start_list[i]) == np.array(prediction < bin_end_list[i]))]
        if len(bin_label) == 0:
            ps.append(0)
            ns.append(0)
            y.append(0)
            continue
        positive_bin_label = bin_label[bin_label == 1]
        negative_bin_label = bin_label[bin_label == 0]
        ps.append(len(positive_bin_label))
        ns.append(len(negative_bin_label))

        y.append(len(positive_bin_label) / len(bin_label))

    plt.figure(fignum, figsize=(8, 5))
    plt.clf()

    #  ax = sns.barplot(x=x, y=y)
    #  ctr = 0
    #  for i, j in zip(x, y):
    #      plt.annotate('{}({}/{})'.format(y[ctr], ps[ctr], ns[ctr]), xy=(i, j))
    #      ctr += 1

    plt.plot(x, y, marker='s', lw=lw, alpha=alpha, color=color)
    ctr = 0
    for i, j in zip(x, y):
        plt.annotate('{:.3f}\n({}/{})'.format(y[ctr], ps[ctr], ns[ctr]), xy=(i, j))
        ctr += 1

    plt.ylabel('Fraction of Positives')
    plt.xlabel('Probability of Model')
    plt.title('Calibration Analysis')
    plt.ylim([0, 1.05])
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_threshold_analysis(fignum, label, prediction, stride=0.05, verbose=False):
    import pandas as pd

    thresholds = []
    fprs       = []
    tprs       = []
    ppvs       = []

    for threshold in np.arange(0, 1, stride):
        if threshold == 0:
            continue
        fp, fn, tp, tn = get_confusion_matrix(label, prediction, threshold, verbose)
        tpr, tnr, ppv, npv, fpr, fnr, fdr = get_metrics(fp, fn, tp, tn, verbose)
        thresholds.append('{:.3f}'.format(threshold))
        fprs.append(fpr)
        tprs.append(tpr)
        ppvs.append(ppv)

    values = np.concatenate((fprs, tprs, ppvs))
    length = len(thresholds)
    thresholds = np.tile(thresholds, 3)
    value_label = np.concatenate((['FPR(1 - Specificity)' for _ in range(length)],
                                  ['TPR(Recall, Sensitivity)' for _ in range(length)],
                                  ['Precision' for _ in range(length)]))
    pd_init = {'value': pd.Series(values),
               'value_label': pd.Series(value_label),
               'threshold': pd.Series(thresholds)}
    data = pd.DataFrame(pd_init)

    plt.figure(fignum, figsize=(10, 5))
    plt.clf()
    ax = sns.barplot(x='threshold', y='value', hue='value_label', data=data, palette='muted')
    plt.yticks(np.arange(0.0, 1.1, 0.2))
    plt.ylim([0, 1.05])
    plt.legend()
    plt.title('Threshold Analysis')
    plt.tight_layout()


def get_area_under_curve(label, prediction):
    fpr, tpr, roc_thr = metrics.roc_curve(label, prediction)
    auc = metrics.auc(fpr, tpr)
    return auc


def plot_roc_curve(fignum, label, prediction, legend, extra_legend='', clf=False, **kwargs):
    lw = kwargs['lw']
    alpha = kwargs['alpha']
    color = kwargs['color']

    fpr, tpr, roc_thr = metrics.roc_curve(label, prediction)
    auc = metrics.auc(fpr, tpr)
    plt.figure(fignum)
    if clf:
        plt.clf()
    plt.plot(fpr, tpr, lw=lw, alpha=alpha, color=color,
             label='{} (AUC={:.3f}{})'.format(legend, auc, extra_legend))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc='lower right')
    plt.tight_layout()

    return fpr, tpr, roc_thr, auc


def get_average_precision(label, prediction):
    return metrics.average_precision_score(label, prediction)


def plot_pr_curve(fignum, label, prediction, legend, extra_legend='', clf=False, **kwargs):
    lw = kwargs['lw']
    alpha = kwargs['alpha']
    color = kwargs['color']

    ap = get_average_precision(label, prediction)
    precision, recall, ap_thr = metrics.precision_recall_curve(label, prediction)
    #  auc = metrics.auc(recall, precision)
    plt.figure(fignum)
    if clf:
        plt.clf()
    plt.step(recall, precision, lw=lw, alpha=alpha, color=color,
             label='{} (AP={:.3f}{})'.format(legend, ap, extra_legend))
             #  label='{} (AP={:.3f}, AUC={:.3f})'.format(legend, ap, auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision recall curve')
    plt.legend(loc='lower left')
    plt.tight_layout()

    return recall, precision, ap_thr, ap

@static_vars(params={}, df_tsne=None)
def plot_tsne(fignum, feature, label, plot_type='scatter', pca_comp=50, clf=False, **kwargs):
    params = {'feature': feature,
              'label': label,
              'pca_comp': pca_comp}

    if plot_tsne.params != params:
        plot_tsne.params = params

        tsne_comp = 2
        lw = kwargs['lw']
        alpha = kwargs['alpha']
        color = kwargs['color']

        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        idx = np.argsort(np.squeeze(label))
        feature = feature[idx]
        label = label[idx]

        #  rndperm = np.random.permutation(feature.shape[0])
        #  feature = feature[rndperm[:1000]]
        #  label = label[rndperm[:1000]]

        feat_col = ['feat'+str(i) for i in range(feature.shape[1])]
        df = pd.DataFrame(feature, columns=feat_col)

        pca = PCA(n_components=pca_comp)
        pca_result = pca.fit_transform(df[feat_col].values)

        pca_col = ['pca'+str(i) for i in range(pca_result.shape[1])]
        df_pca = pd.DataFrame(pca_result, columns=pca_col)

        tsne = TSNE(n_components=tsne_comp, verbose=0)
        tsne_results = tsne.fit_transform(df_pca[pca_col].values)

        tsne_col = ['tsne'+str(i) for i in range(tsne_results.shape[1])]
        df_tsne = pd.DataFrame(tsne_results, columns=tsne_col)
        df_tsne['label'] = label
        df_tsne['label'] = df_tsne['label'].apply(lambda i: str(i))

        plot_tsne.df_tsne = df_tsne

    df_tsne = plot_tsne.df_tsne

    plt.figure(fignum)
    if clf:
        plt.clf()

    if plot_type == 'scatter':
        sns.scatterplot(x='tsne0', y='tsne1', hue='label', data=df_tsne,
                        lw=lw, alpha=alpha, color=color)
    elif plot_type == 'kde':
        sns.kdeplot(np.squeeze(df_tsne.loc[df_tsne['label'] == '0'].filter(['tsne0']).values),
                    np.squeeze(df_tsne.loc[df_tsne['label'] == '0'].filter(['tsne1']).values),
                    shade=True, shade_lowest=False,
                    alpha=alpha)
        sns.kdeplot(np.squeeze(df_tsne.loc[df_tsne['label'] == '1'].filter(['tsne0']).values),
                    np.squeeze(df_tsne.loc[df_tsne['label'] == '1'].filter(['tsne1']).values),
                    shade=True, shade_lowest=False,
                    alpha=alpha)
    else:
        raise ValueError('plot_type must either "scatter" or "kde". yours: {}'.format(plot_type))

    plt.title('t-SNE')
    plt.tight_layout()

def get_metrics_from_sensitivity(label, prediction, sensitivity=0.85):
    import time
    label = np.squeeze(label)
    prediction = np.squeeze(prediction)

    idx = np.argsort(prediction)
    label = label[idx]
    prediction = prediction[idx]

    decision_bound = 1
    step = 0.25
    ctr = 0
    prev = 0

    while True:
        decision_bound -= step
        if decision_bound <= 0.0:
            decision_bound += step
            step *= 0.5
            continue

        prediction_tmp = (prediction >= decision_bound).astype(int)

        cnf_matrix = metrics.confusion_matrix(label, prediction_tmp)
        FP = cnf_matrix[0, 1]
        FN = cnf_matrix[1, 0]
        TP = cnf_matrix[1, 1]
        TN = cnf_matrix[0, 0]

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)

        if TPR > sensitivity:
            decision_bound += step
            step *= 0.5
        if prev == TPR:
            ctr += 1
        else:
            ctr = 0
        if ctr == 20:
            break
        prev = TPR

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Accuracy
    ACC = np.mean(prediction_tmp == label)

    output = {'TPR': TPR,
              'TNR': TNR,
              'PPV': PPV,
              'NPV': NPV,
              'FPR': FPR,
              'FNR': FNR,
              'FDR': FDR,
              'ACC': ACC}

    return output, decision_bound
