# Evaluating and attacking an new detector
## Evaluating against the RAID dataset

In order to evaluate the performance of a given detector on the RAID dataset, we provide the template of the model in the ```src/raid/models/example_model/``` directory:

- ```model.py```:                            Python script containing the main code for loading the model.
- ```model_preprocess.py```:                 Python class that contains the preprocessing done by the model.
- ```model_postprocess.py```:                Python class that contains the postprocessing done by the model.

## Model wrapping and attacking

While the model wrapping allows for testing against the RAID dataset directly, in order to attack the model with the ensemble attack, it is necessary that:
- The model itself doesn't contain any non-differentiable operations in the detection method.
- The preprocessing on the images doesn't contain any non-differentiable transformations, and if there are, it is necessary to substitute them with a differentiable counterpart as noted in ```src/raid/models/example_model/model_preprocess.py```.