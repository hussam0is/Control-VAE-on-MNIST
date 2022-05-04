# Control-VAE-on-MNIST
Implementation of the control-VAE algorithm on MNIST dataset

See the full results and the amazing contribution of Control-VAE [Control VAE On MNIST.pdf](https://github.com/hussam0is/Control-VAE-on-MNIST/blob/main/control%20VAE%20On%20MNIST.pdf)

in [trained_models]() you find: Model-BetaVAE 1.py and Model-ControlVAE kl=9.97.py which contain trained models with the same neural-network structure. each one changes the Beta-value according to the algorithm in the file name on the MNIST dataset

to implement your own VAE network using the control-VAE algorithm you need to change the parameters accordingly in models_def.py and train using train_models.ipynp
