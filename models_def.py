
from audioop import cross
import torch
from torch import Tensor
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from datetime import date, datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print(f"{current_time} models_def.py has been imported successfully!")

torch.manual_seed(0)
plt.rcParams['figure.dpi'] = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# load Dataset
z_dim = 2
epochs = 200
data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', 
               transform=torchvision.transforms.ToTensor(), 
               download=True, train=True),
        batch_size=128,
        shuffle=True, pin_memory = True)

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.encoder_structure = nn.Sequential(
            nn.Linear(784, 400), nn.ReLU(),
            nn.Linear(400, 50), nn.ReLU())
        self.to_mean_logvar = nn.Linear(50, 2*latent_dims)
        
    def reparametrization_trick(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_structure(x)
        mu, log_var = torch.split(self.to_mean_logvar(x),2, dim=-1)
        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return self.reparametrization_trick(mu, log_var)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.decoder_structure = nn.Sequential(
            nn.Linear(latent_dims, 50), nn.ReLU(),
            nn.Linear(50,400), nn.ReLU())
        self.linear4 = nn.Linear(400, 784)

    def forward(self, z):
        z = self.decoder_structure(z)
        z = torch.sigmoid(self.linear4(z))
        return z.reshape((-1, 1, 28, 28))

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)        
        return self.decoder(z)


# VAE
class VAE:
    def __init__(self):
        self.vae_0 = VariationalAutoencoder(z_dim).to(device)
        self.kl_t = 0
        self.param = 0
  
    def _type(self):
        return self.__class__.__name__

    def train(self):
        opt = torch.optim.Adam(self.vae_0.parameters(), lr = 0.001)
        reconstruction_loss = []
        kl_values = []
        beta_values = []
        elbo = []
        ce_loss = nn.CrossEntropyLoss()
        gen_beta = self.gen_beta()
        for epoch in range(epochs):
            epoch_start = True
            for i, (x, y) in enumerate(data_loader):               
                beta_t = next(gen_beta)  
                x = x.to(device) # GPU
                opt.zero_grad()
                x_hat = self.vae_0(x).to(device)
                loss = F.binary_cross_entropy(x_hat, x, reduction='sum') + beta_t*self.vae_0.encoder.kl
                loss.backward()
                opt.step()
                self.kl_t = self.vae_0.encoder.kl.cpu().detach().numpy()/128
                rl_t = loss.item()/128
                reconstruction_loss.append(rl_t)
                kl_values.append(self.kl_t)
                beta_values.append(beta_t)
                elbo.append(-1*(self.kl_t+rl_t))
                if epoch_start:
                    if epoch%1 == 0:
                        print(f'epoch: {epoch+1}/{epochs} batch: #{i+1} beta: {format_4(beta_t)}   kl= {format_4(self.kl_t)} RL= {format_4(rl_t)}')
                    epoch_start=False
        self.reconstruction_loss= reconstruction_loss
        self.kl_values= kl_values
        self.beta_values = beta_values
        self.elbo = elbo
        return self
    
    
# Beta-VAE
class BetaVAE(VAE):
    def __init__(self, beta):
        self.beta = beta
        super().__init__()
        self.param=beta

    def gen_beta(self):
        while True:
            yield self.beta
            
            
# Control-VAE
class ControlVAE(VAE):
    def __init__(self, K_P, K_I, desired_kl, beta_min, beta_max):
        self.K_P = K_P
        self.K_I = K_I
        self.desired_kl = desired_kl
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__()
        self.param = f' kl= {format(desired_kl, ".2f")}'

    def gen_beta(self):
        self.beta = 0
        self.indcr = 0
        while True:
            sample_kl=self.kl_t
            if self.beta<self.beta_max and self.beta>self.beta_min:
                self.indcr = self.indcr-(self.K_I*(self.desired_kl-sample_kl))
            self.beta = self.K_P/(1+np.exp(self.desired_kl-sample_kl))+self.indcr+self.beta_min
            if self.beta<self.beta_min:
                self.beta= self.beta_min
            if self.beta>self.beta_max:
                self.beta = self.beta_max
            yield self.beta


# plot
import matplotlib.pyplot as plt

image_num =1 
def plot(y1,y2, y_label, model1, model2):
    k = 100
    y1  = [np.mean(y1[i:i+k]) for i in range(len(y1)-k)]
    y2  = [np.mean(y2[i:i+k]) for i in range(len(y2)-k)]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Training Steps')
    ax.set_ylabel(y_label)
    ax.set_title(f'{y_label}: {model1} VS {model2}')
    plt.plot(list(range(len(y1))),y1, list(range(len(y2))),y2)
    plt.legend([model1,model2], loc='upper right')
    global image_num
    image_num+=1
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    str_ =  f"MEAN of last {N} {y_label}:\n{model1}: {format(np.mean(y1[-1*N:-1]), '.2f')} \n{model2}: {format(np.mean(y2[-1*N:-1]), '.2f')}"
    plt.text(0.05,0.95, str_ ,{'color':'red','weight':'heavy','size':5},transform=ax.transAxes,bbox=props, verticalalignment='top')
    plt.rcParams.update({'font.size': 6})
    plt.show()

def plot_comparison(model1, model2, last_n =100):
    global N
    N = last_n
    name_a = model1._type()
    name_b = model2._type()
    param_a = model1.param
    param_b = model2.param
    # reconstruction loss:
    plot(model1.reconstruction_loss,model2.reconstruction_loss, f'Reconstruction Loss', f'{name_a} {param_a}', f'{name_b}-{param_b}')
    # kl values
    plot(model1.kl_values,model2.kl_values, f'KL-Divergence', f'{name_a} {param_a}', f'{name_b}-{param_b}')
    # beta values
    plot(model1.beta_values,model2.beta_values, f'Beta(t)', f'{name_a} {param_a}', f'{name_b}-{param_b}')
    # ELBO
    plot(model1.elbo,model2.elbo, f'ELBO', f'{name_a} {param_a}', f'{name_b}-{param_b}')

    
def format_4(num):
    return format(num, '.4f')

def plot_latent(autoencoder, data, num_batches = 2000):
    for i, (x,y) in enumerate(data):
        z = autoencoder.vae_0.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i> num_batches:
            plt.colorbar()
            plt.show()
            break


def get_numbers(data):
    torch.manual_seed(10)
    numbers = list(range(10))
    numbers_x = []
    numbers_y = []
    flag = True
    stop = np.random.randint(1,50)
    for i, (x, y) in enumerate(data):
        if flag == True and i<stop:
            continue
        for j in range(128):
            if y[j] in numbers:
                numbers.remove(y[j])
                numbers_x.append(x[j])
                numbers_y.append(y[j])
            if not numbers:
                break
    return numbers_x, numbers_y


def display_images(reconstructed):
    f, ax = plt.subplots(1,10)
    for i, image in enumerate(reconstructed):
        ax[i].axis('off')
        ax[i].imshow(image.squeeze(), cmap='Blues_r')
    plt.show()


def plot_reconstructed(autoencoder, numbers_x):
    reconstructed= [autoencoder.vae_0(num).detach() for num in numbers_x]
    display_images(reconstructed)
