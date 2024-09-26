Train and analyze different types of models for semi-supervised learning. Contains implementations for logistic regression, linear VAE, convolutional VAE, M1 model and Prediction-constrained VAE. 
Currently supports MNIST and CIFAR-10 datasets.

If you want to run this code, clone the repository into your own directory on the teapot server.

Next, create a virtual environment (most probably in the same directory as your cloned repository). You can also use conda or Docker.

`python -m venv /path_to_your_cloned_repo/env`

`python3.10 -m venv /path_to_your_cloned_repo/env`

Then, activate the environment using `source env/bin/activate`.

Install the requirements using You can install requirements using 

`pip install -r requirements.txt --no-cache-dir`.

You will probably run into to issues installing PyTorch. Visit the Pytorch website to find the distribution matching your system. For the teapot server it should be:

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

Install any other packages that failed using `pip`.

Now you can try training some models.

You should run the scripts from the root directory of the repository. You may have to run this command first:

`export PYTHONPATH="/path_to_your_cloned_repo:$PYTHONPATH"`

Example: `python3 scripts/train_vae.py --config scripts/configs/config-vae-mnist.json`

Make your own config file to edit training. Training is logged to Tensorboard. You can launch Tensorboard in your browser using one of the following commands:

`tensorboard --logdir=logs
python3 -m tensorboard.main --logdir=logs`

You can specify to save a model then load the saved model later.



