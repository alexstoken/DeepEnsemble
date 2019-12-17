# DeepEnsemble
Keras ensembles, made easy.

### Usage
Build or load as many models as you'd like

```
#inception
model1 = models.load_model(
    'models/model1.h5')

#custon (VGG like)
model2 = models.load_model(
    'models/model2.h5')

#InceptionResNet
model3 = models.load_model(
    'models/model3.h5')
```

Next, put them together in an ensemble
```
equip_bin_ensemble = [equip_bin, equip_bin2, equip_bin5]
d = DeepEnsemble(equip_bin_ensemble, pos_label=1)
print(f'This is a {d.model_type} model.\n')
```

Always use DeepEnsemble with your __validation__ set.

```
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size_ = 1871

val_gen = val_datagen.flow_from_directory('learning_ready_data/val',  batch_size=batch_size_, classes=[
                                        'equip', 'no_equip'],  target_size=(256, 384), class_mode='binary')

samples, sample_classes = val_gen.next()
```

Evaluate your ensemble on the validation set, using `voting` or `average` and with an option to
find and use an `optimal threshold`. 

```
eval_as = 'average'
print(f'Evaluating model with {eval_as}')
d.evaluate(samples, sample_classes, eval_type=eval_as, optimal=True)
```

Look at various aspects of model performance
```
print(d._confusion_matrix)
print(f'Model accuracy scores {d.acc_scores}')
d.plot_confusion(show=True)
d.plot_roc(show=True)
```
Use two methods to find an weighted average of models that performs best on the validation set. You can use [differential evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) with `.find_weighted_avg()` or a meta-learner, a one layer neural network with the output of the sub-models (probability scores) as the input to the meta-learner. The meta-learner can be accessed with `.train_meta_learner()`. 

```
print('Finding weighted average')
d.find_weighted_avg()
print('Training meta-learner')
d.train_meta_learner()
```
Finally, predict using the ensemble with
``` 
d.predict(samples)
```

### Contributions

Contributions are welcome! Please create an issue or a PR to collaborate on this library. 
