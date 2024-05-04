Train and analyze different types of models for semi-supervised learning. Contains implementations for logistic regression, linear VAE, convolutional VAE, M1 model and Prediction-constrained VAE. 
Currently supports MNIST and CIFAR-10 datasets.

Example: `python3 scripts/train_vae.py --config scripts/configs/config-vae.json`

You should run the scripts from the root directory (`semisupervisedVAE`). You may have to run this command first:
`export PYTHONPATH="/semisupervisedVAE:$PYTHONPATH"`
