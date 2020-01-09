"""
Deep Ensemble class

Created:
Aug 1 2019 
Alex Stoken

Updated:
sDec 19 @019

Class designed to make ensembles of keras models easy to train, test, and modify.

BINARY MODELS ONLY as of Aug 15 2019.

#TODO support for multiclass/regression models
"""

from numpy.linalg import norm
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers, callbacks, activations
from tensorflow.keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import utils
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import differential_evolution
import math
import pickle


class DeepEnsemble(object):
    """Manages and evaluates ensembles of Keras models.
        
    Attributes:
        model_list [list]: keras model instances with .predict() method
        metrics [dict]: dictionary of useful metrics (AUC, f1, confusion matrix)
        thresh [float]: classification threshold 
        true_labels [list]: test labels in a list 
        test_data [list]: test data samples in a list
        acc_scores [dict]= accuracy of submodels and ensemble model
        ensemble_probas [list]: predicted probabilities from ensemble model
        class_preds [list]: class label predictions from ensemble model
        model_type [str]: binary or categorical
        weights [list]: weights for models in ensemble (default: 1/num_models for each model)
    """

    def __init__(self, model_list = [], pos_label = None):
        self.model_list = model_list
        self.pos_label = pos_label
        self.weights = np.array([1.0/len(self.model_list)
                                 for _ in self.model_list])

        self._check_list()

    def _check_list(self):
        """Checks that ensemble only contains Keras models
        
        Raises:
            TypeError: Error if not all models are Keras models
        """
        if self.model_list[0].output_shape[-1] ==1:
            self.model_type = 'binary'
            if not self.pos_label:
                self.pos_label = int(input("Enter numerical class label for positive class: "))
        else:
            self.model_type = 'categorical'
            raise TypeError('DeepEnsemble currently only supports binary models. Check soon for updates.')
        for model in self.model_list:
            if not isinstance(model, Model):
                raise TypeError('All models must be Keras functional or sequential models')

    def _compute_accuracy(self, preds, true_labels):
        """Internal method for calculating accuracy given a threshold
        
        Arguments:
            preds {list/ndarray} -- list of predicted class probabilities (binary only)
            true_labels {list/ndarray} -- ground/true class labels
        
        Returns:
            float -- accuracy score
        """
        correct = 0

        for p, true_label in zip(preds, true_labels):
            if p > self.thresh:
                pred_label = 1
            else:
                pred_label = 0

            if pred_label == true_label:
                correct = correct + 1

        return correct/len(preds)
    
    def _compute_multiclass_accuracy(self, preds, true_labels, top_n= 1):
        pass

    def _compute_regression_mse(self, preds, true_vals):
        pass


    def _compute_metrics(self):
        """Computes various helpful ML metrics for the ensemble model

        Currently implemented metrics are:
        -confusion matrix
        -AUC score
        -F1 score

        All scores can be accessed via self.metrics

        Arguments:
            None
        
        Returns: 
            None

        """
        from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

        self.metrics = {}

        self._confusion_matrix = confusion_matrix(self.true_labels, self.class_preds)
        self.metrics['confusion matrix'] = self._confusion_matrix

        self._rocauc = roc_auc_score(self.true_labels, self.ensemble_probas)
        self.metrics['AUC'] = self._rocauc

        self._f1score = f1_score(self.true_labels, self.class_preds)
        self.metrics['F1 score'] = self._f1score

    def _compute_optimal_thresh(self, truth_labels, predicted_probas):
        """Calcualtes the threshold that gives the highest True Positive Rate - False Positive Rate using sklearn backend. 

        In essence, this is threshold (regardless of class) with the greatest difference
        between Correct Predictions and Incorrect Predictions 
        NOTE: it DOES NOT necessarily maximize correct predictions OR minimize incorrect predictions
        
        Arguments:
            truth_labels {List(int)} -- numerically encoded ground truth labels
            predicted_probas {List(float)} -- probability predictions from classifier
        
        Returns:
            [float] -- Optimal decision threshold value
        """
        fpr, tpr, thresh_ = roc_curve(
            truth_labels, predicted_probas, pos_label=self.pos_label)
        optimal_idx = np.argmax(np.abs(tpr-fpr))
        return thresh_[optimal_idx]

    def add_models(self, model_to_add):
        """Add model to ensemble 
        
        Arguments:
            model_to_add {Keras Model or list(Keras Model)} -- Model or list of models to add to ensemble
        """
        if type(model_to_add) == list:
            self.model_list.extend(model_to_add)
            self._check_list()
        else:
            self.model_list.append(model_to_add)
            self._check_list()
        self.weights = np.array([1.0/len(self.model_list)
                                 for _ in self.model_list])

    def clear_models(self):
        """Clears all models from ensemble 
        TODO: remove all attributes from class as well
        """
        self.model_list = []
        self.weights = np.array([1.0/len(self.model_list)
                                 for _ in self.model_list])

    def evaluate(self, test_data, test_labels, eval_type ='average', thresh = 0.5, optimal = False, weighted = False) -> dict:
        """Evaluates accuracy and other metrics of ensemble using either average or voting system. 

        Generates predicted probabilities and predicted class labels for ensemble. 
        
        Arguments:
            test_data {[type]} -- Test data samples
            test_labels {[type]} -- Ground truth labels for test data
        
        Keyword Arguments:
            eval_type {str} -- Average submodel probas or use voting to calculate ensemble preds (default: {'average'})
            thresh {float} -- classification threshold (default: {0.5})
            optimal {bool} -- use TPR-FPR to find optimal threshold (default: {False})
        
        Raises:
            ValueError: Eval type must be 'average' or 'voting'
        
        Returns:
            dict -- Accuracy scores for all submodels and ensemble model
        """
        print('running')
        from scipy.stats import mode
        
        self.thresh = thresh
        self.true_labels = test_labels
        self.test_data = test_data
        self.acc_scores = {} #initialize empty accuracy dict
        self.preds_list = []
        for i, model in enumerate(self.model_list):
            print(f'running model {i}')
            preds = model.predict(test_data)
            self.preds_list.append(preds)
            self.acc_scores[f'model{i}'] = self._compute_accuracy(preds, test_labels)

        self.ensemble_probas = sum(self.preds_list) / len(self.preds_list)
        

        #optimal threshold is defined as the threshold which 
        #creates the maximum value for (True Positive Rate - False Positive Rate)
        #In essence, this is threshold (regardless of class) with the greatest difference
        #between Correct Predictions and Incorrect Predictions 
        #NOTE: it DOES NOT necessarily maximize correct predictions OR minimize incorrect predictions
        if optimal:
            self.thresh = self._compute_optimal_thresh(truth_labels=self.true_labels, predicted_probas=self.ensemble_probas)

        if eval_type == 'average':
            self.acc_scores['ensemble'] = self._compute_accuracy(self.ensemble_probas, test_labels)
        elif eval_type == 'voting':
            vote_list = []
            for m in range(len(self.preds_list)):
                if optimal:
                    sub_thresh = self._compute_optimal_thresh(truth_labels=self.true_labels, predicted_probas=self.preds_list[m])
                    print(sub_thresh)
                else:
                    sub_thresh = self.thresh

                votes = [1 if p > sub_thresh else 0 for p in self.preds_list[m]]
                vote_list.append(votes)

            vote_list = np.array(vote_list)
            ens_vote = mode(vote_list)[0][0]

            self.acc_scores['ensemble'] = self._compute_accuracy(
                ens_vote, test_labels)
        else:
            raise ValueError("eval_type must be 'average' or 'voting'.")
            
        self.class_preds = [1 if p > self.thresh else 0 for p in self.ensemble_probas]
        
        if weighted == True:
            self.find_weighted_avg()

        self._compute_metrics()


        with open('submodel_preds', 'wb') as submodel_output_file:
            pickle.dump(self.preds_list, submodel_output_file)

        
        return self.acc_scores

    """def train_weighted_avg(self, training_data, training_labels, epochs = 10, batch_size= None):
        self.top = models.Sequential()
        self.top.add(layers.Dense(1, input_dim=(
            len(self.model_list)), activation='sigmoid'))

        self.top.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=['acc'])
        callback_list = [callbacks.EarlyStopping(patience=3, restore_best_weights=True)]


        self.top.fit(training_data, training_labels, epochs, callbacks = callback_list, validation_split = .15)
        #TODO make training data in the form of ([submodel_preds], truth)
        return self.top
    """

    def find_weighted_avg(self, validation_split= 0, validation_data: (list, list) = None):
        """Use the differential evolution optimization algorithm from scipy to find optimal weights for ensemble.

        Differential evolution finds a global minimum in the search space (0,1) for each weight. Weights are
        normalized following the calculation. For more info, 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

        
        Keyword Arguments:
            validation_split {float} -- Portion of samples to be held out for validation (default: {0})
            validation_data {tuple(list, list)} -- Tuple of (samples, labels) for validation
        """

        #if validation_split != 0:
        weighted_data = np.array(self.preds_list).transpose()[0]
        weighted_data = weighted_data[:round(len(weighted_data)*(1-validation_split))]
        """elif validation_data is not None:
            val_preds_list = []
            for model in self.model_list:
                preds = model.predict(validation_data[0])
                val_preds_list.append(preds)

            weighted_data = np.array(val_preds_list).transpose()[0]
        else:
            raise('Need validation data to run proper optimization')
        """
        
        bound_w = [(0.0, 1.0) for _ in self.model_list]
        # arguments to the loss function
        search_arg = (weighted_data,)

        # global optimization of ensemble weights
        result = differential_evolution(
            self._weighted_avg_loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)

        # get the chosen weights
        self.weights = self._normalize(result['x'])
        print('Optimized Weights: %s' % self.weights)
        # evaluate chosen weights
        score = self._weighted_eval(weighted_data, self.weights)
        print('Optimized Weights Score: %.5f' % score)

        return None

    def _weighted_eval(self, preds, weights):
        """Find weighted average accuracy of ensemble
        
        Arguments:
            preds {list(float)} -- predicted probabilities
            weights {list(float)} -- weights of models
        
        Returns:
            float -- accuracy score for weighted ensemble
        """
        weighted_preds = np.dot(preds, weights)

        self.acc_scores['weighted_ensemble'] = self._compute_accuracy(weighted_preds, self.true_labels)

        return self.acc_scores['weighted_ensemble']

    def _normalize(self, vec):
        """Normalize a vector to 1
        
        Arguments:
            vec {ndarray or array-like} -- unnormalized vector
        
        Returns:
            ndarray -- normalized vector
        """
        # calculate l1 vector norm
        result = norm(vec, 1)
        # check for a vector of all zeros
        if result == 0.0:
            return vec
        # return normalized vector (unit norm)
        return vec / result

    def _weighted_avg_loss_function(self, weights, preds):
        """Loss function for differential evolution (used to find optimal weights for weighted ensemble)
        
        Arguments:
            weights {List(float)} -- Weights of each model in ensemble
            preds {List(float)} -- Ensemble probability predictions
        
        Returns:
            [float] -- Loss
        """
        # normalize weights
        normalized = self._normalize(weights)
        # calculate error rate
        return 1.0 - self._weighted_eval(preds, normalized)

    def train_meta_learner(self, val_data, val_labels):

        for i, m in enumerate(self.model_list):
            for layer in m.layers:
                layer.trainable = False
                layer.name = f'ens_{i}_{layer.name}'

        ens_inputs = [m.input for m in self.model_list]
        ens_output = [m.output for m in self.model_list]

        merge = layers.merge.concatenate(ens_output)
        hidden = layers.Dense(10, activation='relu')(merge)
        output = layers.Dense(
            self.model_list[0].output_shape[-1], activation='softmax')(hidden)

        meta = Model(inputs=ens_inputs, outputs=output)

        meta.compile(loss=self.model_type + '_crossentropy',
                     optimizer='adam', metrics=['accuracy'])

        callback_list = [callbacks.EarlyStopping(patience=15, restore_best_weights=True), callbacks.ModelCheckpoint(
            filepath='./model_ep{epoch:02d}-loss{val_loss:.2f}.hdf5')]

        meta_hist = meta.fit(self.test_data, self.true_labels, epochs = 150, callbacks = callback_list, validation_split = .2)

        utils.save_model(meta, meta_hist, 'meta_equip')

        return meta

    def predict(self, samples, confidence = False):
        """Generate class predictions for input samples
        
        Arguments:
            samples {List(samples)} -- List of samples of the same type as the training data, to be predicted on 
            confidence {List(float)} -- List of confidence levels in prediction
        Returns:
            List(int) -- List of class predictions
        """
        output_preds = []
        for model in self.model_list:
            output_preds.append(model.predict(samples))
        
        output_preds = np.array(output_preds)[:,:,0]

        self.weighted_ensemble_probas = np.dot(self.weights, output_preds )
        self.weighted_class_preds = [
            1 if p > self.thresh else 0 for p in self.weighted_ensemble_probas]
        
        if confidence == True:
            return self.weighted_class_preds, self.weighted_ensemble_probas
        else: return self.weighted_class_preds


    def plot_roc(self, save_name = None, optimal = True, show = True):
        """Generates ROC curve for ensemble model and submodels

        Keyword Arguments:
            show {bool} -- print plot to screen (default: {False})
            optimal {bool} -- plot a dot for the optimal threshold (default: {True})
            save_name {str} -- file path/name WITH filetype to save plot
        
        Returns:
            plt.ax -- ax object containing ROC curve
        """
        
        fpr, tpr, thresh_ = roc_curve(
            self.true_labels, self.ensemble_probas, pos_label=self.pos_label)
  
        fpr_list = []
        tpr_list = []

        try:
            for preds in self.preds_list:
                fpr_, tpr_, temp_thresh_ = roc_curve(self.true_labels, preds, pos_label = self.pos_label)
                fpr_list.append(fpr_)
                tpr_list.append(tpr_)
        except NameError:
            print("You must run .evaluate() first to generate predictions before plotting a ROC curve.")

        auc = roc_auc_score(self.true_labels, self.ensemble_probas)
        optimal_idx = np.argmax(np.abs(tpr-fpr))
        optimal_thresh = thresh_[optimal_idx]

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='orange', label=f'ROC (area={auc:.4f})')
        
        for fpr_, tpr_ in zip(fpr_list, tpr_list):
            ax.plot(fpr_, tpr_, '--' ,label = f'Submodel ROC')

        ax.set_title('ROC for Ensemble and Submodels')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')

        ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        ax.plot(fpr[optimal_idx],tpr[optimal_idx] , color='black', marker = 'o', markersize = '5',
                label=f'Optimal Threshold = {optimal_thresh:.4f}')

        ax.legend()
        if show == True:
            plt.show()
        if save_name:
            fig.savefig(save_name)
        return ax

    def plot_confusion(self, save_name=None, show=True):
        """Wrapper for utils plot_confusion_matrix function 
        
        Keyword Arguments:
            show {bool} -- print plot to screen (default: {False})
            save_name {str} -- file path/name WITH filetype to save plot
        
        Returns:
            plt.ax -- ax object containing confusion matrix
        """
        fig, ax = utils.plot_confusion_matrix(self._confusion_matrix)
        if show ==True:
            plt.show()
        if save_name:
            fig.savefig(save_name)
        return ax

    def plot_precision_recall(self, save_name = None, show = True ):
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, thresholds = precision_recall_curve(self.true_labels, self.weighted_ensemble_probas, pos_label=self.pos_label)

        avg_precision = average_precision_score(self.true_labels, self.weighted_ensemble_probas, pos_label= self.pos_label)

        fig, ax = plt.subplots()
        ax.plot(recall, precision, label='Ensemble')


        ax.axhline(y=sum(self.true_labels == 0)/len(self.true_labels),
                    color='r', linestyle = 'dashed', label='Baseline')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision Recall Curve (AUC = {avg_precision:.4f})')
        ax.grid(True, which='major', axis='both', linestyle='-')
        ax.minorticks_on()
        ax.grid(True, which='minor', axis='both', linestyle=':')
        ax.legend()
            

        if show == True:
            plt.show()
        if save_name:
            fig.savefig(save_name)
        return ax

    def plot_threshold(self, save_name = None, show = True):
        """[summary]
        
        Keyword Arguments:
            save_name {[type]} -- [description] (default: {None})
            show {bool} -- [description] (default: {True})
        """
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(self.true_labels, self.weighted_ensemble_probas, pos_label=self.pos_label)
        
        fig, ax = plt.subplots()
        ax.plot(thresholds, precision, label='Precision')
        ax.plot(thresholds, recall, label='Recall')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Threshold')
        ax.set_ylabel(f'% of Dataset')
        ax.set_title(f'Precision/Recall at Thresholds')
        ax.grid(True, which='major', axis='both', linestyle='-')
        ax.minorticks_on()
        ax.grid(True, which='minor', axis='both', linestyle=':')
        ax.legend()
    

        if show == True:
            plt.show()
        if save_name:
            fig.savefig(save_name)
        return ax
if (__name__ =='__main__'):
    #=================================================================
    # TEST AREA
    #=================================================================
    print('loading models...')
    K.clear_session()
 
    #inception
    equip_bin = models.load_model(
        'models/equip_transfer_tuning_07-24-2019_17_32/equip_transfer_tuning_07-24-2019_17_32.h5')
    
    #custon (VGG like)
    equip_bin2 = models.load_model(
        'models/equip_cnn_07-30-2019_13_47/equip_cnn_07-30-2019_13_47.h5')

    #Inception
    #equip_bin4 = models.load_model(
    #    'models/equip_transfer_tuning_07-31-2019_18_56/equip_transfer_tuning_07-31-2019_18_56.h5')
    #InceptionResNet
    equip_bin5 = models.load_model(
        'models/equip_transfer_tuning_08-06-2019_17_24/equip_transfer_tuning_08-06-2019_17_24.h5')

    #equip_bin_ensemble = [equip_bin, equip_bin2, equip_bin4, equip_bin5]
    equip_bin_ensemble = [equip_bin, equip_bin2, equip_bin5]
    #equip_bin_ensemble = [equip_bin, equip_bin2]

    val_datagen = ImageDataGenerator(rescale=1./255)

    batch_size_ = 1871

    print('\nGenerating validation data...')
    val_gen = val_datagen.flow_from_directory('learning_ready_data/val',  batch_size=batch_size_, classes=[
                                            'equip', 'no_equip'],  target_size=(256, 384), class_mode='binary')

    samples, sample_classes = val_gen.next()


    print(f'Making ensemble of models {equip_bin_ensemble}')
    d = DeepEnsemble(equip_bin_ensemble, pos_label=1)
    print(f'This is a {d.model_type} model.\n')

    eval_as = 'average'
    print(f'Evaluating model with {eval_as}')
    d.evaluate(samples, sample_classes, eval_type=eval_as, optimal=True)

    #print(d._confusion_matrix)
    #print(f'Model accuracy scores {d.acc_scores}')
    #d.plot_confusion(show=True)
    #d.plot_roc(show=True)
    print('Finding weighted average')
    d.find_weighted_avg()

    print('\nGenerating testing data...')
    test_gen = val_datagen.flow_from_directory('learning_ready_data/test',  batch_size=1874, classes=[
        'equip', 'no_equip'],  target_size=(256, 384), class_mode= 'binary')

    test_samples, test_labels = test_gen.next()


    outputs = d.predict(test_samples)
    print(f'Test set accuracy: {np.sum(outputs == test_labels)/batch_size_}')
    print(d.plot_roc(show = True))
    print(d.plot_confusion(show = True))
    print(d.thresh)
    print(d.acc_scores)
    print('Done')

