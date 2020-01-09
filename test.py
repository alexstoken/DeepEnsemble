from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, Model

import utils
from deep_ensemble import *
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
    'equip', 'no_equip'],  target_size=(256, 384), class_mode='binary')

test_samples, test_labels = test_gen.next()

outputs = d.predict(test_samples)
print(f'Test set accuracy: {np.sum(outputs == test_labels)/batch_size_}')
print(d.plot_roc(show=True))
print(d.plot_confusion(show=True))
print(d.thresh)
print(d.acc_scores)
print('Done')
