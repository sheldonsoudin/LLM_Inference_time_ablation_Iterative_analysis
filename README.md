# LLM_Inference_time_ablation_Iterative_analysis

 LLM Inference-time ablation: Iterative analysis of robustness / sensitivity


In progress: due april 19th for comp432:machine learning
model.py : in progress 
data.py : todo 
train.py : todo 

Abstract


In this project, I propose to train a transformer based large language model, and
investigate the parameter sensitivity of the model performance to targeted weight ablations,
and determine whether certain parameter subsets (structural groupings) disproportionately
affect chosen benchmark accuracy. In other words, I will investigate whether specific subsets
of parameters disproportionately influence benchmark performance, and whether such
sensitivities are task-dependent.

1. A from-scratch trained transformer model (~300 million parameters)
2. A pretrained model of comparable size (GPT-Neo 350M and/or Pythia 410M)

I would like to explore if there is a way to improve the model against benchmarks
without gradient based retraining, using only inference-time parameter manipulation The
broader intent of this project is to explore how general purpose models can be adapted to
specific use cases by identifying and exploiting structurally important parameters rather than
full retraining.

See proposal for all information.


