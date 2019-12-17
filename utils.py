#file handling
import os
import datetime
import shutil
import fnmatch
import pathlib

#plotting
import matplotlib.image as image
import matplotlib.pyplot as plt

#ML utils
from tensorflow.keras.utils import plot_model
import numpy as np

def save_model(model, history, dir_name, show_plots=True):
    """Save Keras model, model diagram, history, loss and accuracy plots all in one folder
    
    Arguments:
        model {tf.keras.Model} -- Trained Keras model
        history {tf.keras.callbacks.History} -- Stores the data about the Keras model
        dir_name {str} -- Name to save file under
    """
    import pickle
    current_time = datetime.datetime.now().strftime("%m-%d-%Y_%H_%M")
    new_dir = dir_name + '_' + current_time

    try:
        os.mkdir(new_dir)
    except Exception as e:
        print(e)

    move_dir('.', new_dir, pattern='model_ep*')
    model_save_name = new_dir + '.h5'

    try:
        model.save(new_dir + '/' + model_save_name)
    except Exception as e:
        print(f'Could not save model with exception: {e}')
    try:
        plot_model(model, to_file=new_dir + '/' + dir_name+'_' +
                   current_time+'.png', show_shapes=True, show_layer_names=True)
    except Exception as e:
        print(f'Could not plot model with exception: {e}.')

    plot_history(history, metric='all',
                 save_path=new_dir, show_plot=show_plots)

    with open(new_dir + '/' + dir_name+'_' + current_time+'_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    if not os.path.isdir('./models'):
        os.mkdir('models')
    shutil.move(new_dir, 'models')

    return 'models/' + new_dir + '/' + model_save_name


def plot_history(history_list, metric: str, save_path=None, logo_path=None, show_plot=True):
    """Plot the Model Loss for a Keras Classifier
    
    Arguments:
        List([Keras.callbacks.history]) {List of Keras model history objects} -- list of Keras histories which contain information about model performance
        Metric {str} -- choose a metric to plot. "all" to plot all metrics
    
    Keyword Arguments:
        save_path {str} -- path to save file (default: {None})
        logo_path {str} -- path to jpg logo (default: {None})
        show_plot {bool} -- print the plot (default: {True})

    Returns:
        pyplot.ax -- axis containing accuracy plot

    #TODO take in history list, single history object, or history.history dict
    #TODO optimize and clean up logic/repetitiveness
    """

    #type check the history to prepare input object for plotting

    """if isinstance(history_list, (tf.keras.callbacks.History, dict)):
        history_list = [history_list]
    """

    from scipy import stats

    #if more than one history is added make them into a list
    if not isinstance(history_list, list):
        history_list = [history_list]

    #make a list of all of the metrics in the model from the keys of the dict
    if isinstance(history_list[0], dict):
        metric_list = list(history_list[0].keys())
    elif isinstance(history_list[0], tf.keras.callbacks.History):
        metric_list = list(history_list[0].history.keys())
    else:
        metric_list = list(history_list[0].history.keys())

    if metric == "all":
        for metric in metric_list:
            if 'val' not in metric:
                plot_history(history_list, metric, save_path=save_path,
                             logo_path=logo_path, show_plot=show_plot)
        return

    fig, ax = plt.subplots()

    if logo_path:
        logo = image.imread(logo_path)
        ax.imshow(logo, aspect='auto', extent=(.05, .25, .05, .25),
                  alpha=0.1, zorder=-1, transform=ax.transAxes)

    train = []
    val = []

    if isinstance(history_list[0], dict):
        metric_list = list(history_list[0].keys())
        for h in history_list:
            for m in h[metric]:
                train.append(m)
            try:
                for v in h[f'val_{metric}']:
                    val.append(v)
            except Exception as e:
                print(f'Could not plot validation data with exception {e}')

    elif isinstance(history_list[0], tf.keras.callbacks.History):
        metric_list = list(history_list[0].history.keys())
        for h in history_list:
            for m in h.history[metric]:
                train.append(m)
            try:
                for v in h.history[f'val_{metric}']:
                    val.append(v)
            except Exception as e:
                print(f'Could not plot validation data with exception {e}')

    else:
        metric_list = list(history_list[0].history.keys())
        for h in history_list:
            for m in h.history[metric]:
                train.append(m)
            try:
                for v in h.history[f'val_{metric}']:
                    val.append(v)
            except Exception as e:
                print(f'Could not plot validation data with exception {e}')

    slope = stats.linregress(np.arange(0, len(train), 1), train)[0]

    if slope > 0:
        ax.plot(train, label=f'Train (max = {max(train):.4f})')

        ax.plot(np.argmax(train), np.max(train),
                marker='o', color='black', markersize='3')
        if len(val) > 0:
            ax.plot(val, label=f'Validation (max = {max(val):.4f})')
            ax.plot(np.argmax(val), np.max(val), marker='o',
                    color='black', markersize='3')

    else:
        ax.plot(train, label=f'Train (min = {min(train):.4f})')

        ax.plot(np.argmin(train), np.min(train),
                marker='o', color='black', markersize='3')

        if len(val) > 0:
            ax.plot(val, label=f'Validation (min = {min(val):.4f})')
            ax.plot(np.argmin(val), np.min(val), marker='o',
                    color='black', markersize='3')

    ax.set_title('Model ' + metric)
    ax.set_ylabel(metric)
    ax.set_xlabel('Epoch')
    ax.grid(True, which='major', axis='both', linestyle='-')
    ax.minorticks_on()
    ax.grid(True, which='minor', axis='both', linestyle=':')
    ax.legend()

    if save_path:
        plt.savefig(save_path+'/' + metric + '.pdf')

    if show_plot == True:
        plt.show()

    return ax


def move_dir(src: str, dst: str, pattern: str = '*'):
    """Move an entire folder from one dir to another
    
    Arguments:
        src {str} -- path to source (can be '.')
        dst {str} -- path to destination (full or relative)
    
    Keyword Arguments:
        pattern {str} -- uses fnmatch pattern matching (default: {'*'})
    """
    if not os.path.isdir(dst):
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
    for f in fnmatch.filter(os.listdir(src), pattern):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))
