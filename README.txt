This is a repository to predict VHEE dose distributions using a deep learning model (E-DoTA) based on the patient geometry, the general beam shape 
and the particle energy.

This repository is created for the purpose of a master thesis, which can be found here: http://resolver.tudelft.nl/uuid:6b196ab5-19fd-4d37-9b75-f9addaca16f4



Usage:

- Create a venv using the requirements.txt file.

The patient geometry model inputs and the ground truth dose distributions should already be created. 
The general beam shape can be created using the utils/flux_generator.py file.

To Train the model run trainE-DoTA.py (E-DoTA) or TrainAblation.py (E-DoTA-Conv).

The Data generator is defined under src/DataGenerator.py.

The model is defined in src/model.py (E-DoTA) and src/modelAblation.py (E-DoTA-Conv). The building blocks of the model can be found under src/Blocks.py

The model predictions can then be generated using utils/inference.py.

Evaluation and plotting of the results can be done using utils/plot.py and utils/metrics.py.




