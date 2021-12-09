# PoE
PoE is the abbreviation of 'Power of Explanations'. 
The aim of the project is to use explanation methods to correct undesired behaviors of models in the training phase.

Models can be over-sensitive to particular words in many cases, e.g. imbalance distribution of labels in a given dataset.
And thus, can easily make problematic decision based on incorrect observations.
By having this assumption, we can correct the model and encourage them not to focus on the 'incorrect' observation even though we have no knowledge about how it should behave. 
