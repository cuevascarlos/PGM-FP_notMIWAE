import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist
from torch.distributions.independent import Independent


class Encoder(nn.Module):
    def __init__(self, input_dim, layer_sizes, latent_dim, activation):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        layers = [nn.Linear(input_dim, layer_sizes[0]), activation]
        for i in range(len(layer_sizes) - 1):
            layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), activation])

        self.layers = nn.ModuleList(layers)        
        self.mu_head = nn.Linear(layer_sizes[-1], latent_dim)
        self.std_head = nn.Linear(layer_sizes[-1], latent_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        mu = self.mu_head(x)
        log_std = self.std_head(x)

        return mu, torch.clamp(log_std, min = -10, max = 10)

class ImputationDecoder(nn.Module):
    def __init__(self, latent_dim, layer_sizes, output_dim, activation):
        super(ImputationDecoder, self).__init__()
        layers = [nn.Linear(latent_dim, layer_sizes[0]), activation]
        for i in range(len(layer_sizes) - 1):
            layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), activation])

        self.layers = nn.ModuleList(layers)
        self.mu_head = nn.Linear(layer_sizes[-1], output_dim)
        self.std_head = nn.Sequential(nn.Linear(layer_sizes[-1], output_dim), nn.Softplus())

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        mu = self.mu_head(z)
        std = self.std_head(z)
        return mu, std
    
class MaskDecoder(nn.Module):
    def __init__(self, latent_dim, layer_sizes, output_dim, activation):
        super(MaskDecoder, self).__init__()
        layers = [nn.Linear(latent_dim, layer_sizes[0]), activation]
        for i in range(len(layer_sizes) - 1):
            layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), activation])

        self.layers = nn.ModuleList(layers)
        self.logits_head = nn.Linear(layer_sizes[-1], output_dim)

    def forward(self, eta):
        for layer in self.layers:
            eta = layer(eta)
        logits = self.logits_head(eta)
        return logits

class notMIWAE(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_layer_sizes, dec_layer_sizes, activation = nn.Tanh(), missing_process='selfmask'):
        super().__init__()
        self.encoder = Encoder(input_dim, enc_layer_sizes, latent_dim, activation)
        self.decoder = ImputationDecoder(latent_dim, dec_layer_sizes, input_dim, activation)
        self.activation = activation
        self.missing_process = MissingProcess(input_dim, missing_process)
        self.prior = dist.normal.Normal(loc = 0., scale = 1.)

    def forward(self, x, s, n_samples = 1, return_x_samples = False):
        q_mu, q_log_std2 = self.encoder(x)
        law_z_given_x = dist.normal.Normal(loc = q_mu, scale = torch.exp(q_log_std2 * 0.5))

        # Sampling and computing log_probs
        z_samples = law_z_given_x.rsample((n_samples,)) # (n_samples, batch, n_latent)
        log_prob_z_given_x = law_z_given_x.log_prob(z_samples).sum(dim=-1) # (n_samples, batch))

        # Transposing
        z_samples = z_samples.transpose(0,1) # (batch, n_samples, n_latent)
        log_prob_z_given_x = log_prob_z_given_x.transpose(0,1) # (batch, n_samples)

        # Prior
        log_prob_z = self.prior.log_prob(z_samples).sum(dim=-1) # (batch, n_samples)

        p_mu, p_std = self.decoder(z_samples)
        law_x_given_z = dist.normal.Normal(loc = p_mu, scale = p_std)
        x_samples = law_x_given_z.rsample()
        log_prob_x_given_z = (law_x_given_z.log_prob(x.unsqueeze(1)) * s.unsqueeze(1)).sum(dim=-1) # (batch, n_samples)

        mixed_x_samples = x_samples * (1-s).unsqueeze(1) + (x*s).unsqueeze(1)
        logits = self.missing_process(mixed_x_samples)
        law_s_given_x = dist.bernoulli.Bernoulli(logits = logits)
        log_prob_s_given_x = law_s_given_x.log_prob(s.unsqueeze(1)).sum(dim=-1) # (batch, n_samples)

        if return_x_samples:
            return q_mu, q_log_std2, p_mu, p_std, log_prob_z_given_x, log_prob_z, log_prob_x_given_z, log_prob_s_given_x, x_samples

        return q_mu, q_log_std2, p_mu, p_std, log_prob_z_given_x, log_prob_z, log_prob_x_given_z, log_prob_s_given_x


def not_miwae_loss(log_prob_z_given_x, log_prob_z, log_prob_x_given_z, log_prob_s_given_x, device="cpu"):
    # print(log_prob_z.shape)
    aux = torch.logsumexp(-log_prob_z_given_x + log_prob_z + log_prob_x_given_z + log_prob_s_given_x, dim = 0) - torch.log(torch.tensor([log_prob_z.shape[0]], device=device))
    # print(aux.shape)
    # print("loss")
    loss =  - (torch.mean(torch.logsumexp(-log_prob_z_given_x + log_prob_z + log_prob_x_given_z + log_prob_s_given_x, dim = 0) - torch.log(torch.tensor([log_prob_z.shape[1]], device=device))))
    # print(loss.shape)
    return loss


class DualVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_layer_sizes, dec_layer_sizes, activation = nn.Tanh()):
        super(DualVAE, self).__init__()

        # Encoder and decoder for the generative process
        self.encoder_gen = Encoder(input_dim, enc_layer_sizes, latent_dim, activation)
        self.imputation_decoder = ImputationDecoder(latent_dim, dec_layer_sizes, input_dim, activation)

        # Encoder and decoder for the mask process
        self.encoder_mask = Encoder(input_dim, enc_layer_sizes, latent_dim, activation)
        self.mask_decoder = MaskDecoder(latent_dim, dec_layer_sizes, input_dim, activation)

        # Print model
        #print(self.encoder_gen)
        #print(self.imputation_decoder)
        #print(self.encoder_mask)
        #print(self.mask_decoder)
        # Prior distribution (standard Normal)
        self.prior = dist.Normal(loc=0., scale=1.)

    def forward(self, x, s, n_samples=1, return_x_samples=False, only_imputation=False):
        # Generative process
        q_mu_z, q_log_std_z = self.encoder_gen(x)
        law_z_given_x = dist.Normal(loc=q_mu_z, scale=torch.exp(0.5 * q_log_std_z))
        z_samples = law_z_given_x.rsample((n_samples,))  # (n_samples, batch, latent_dim)
        log_prob_z_given_x = law_z_given_x.log_prob(z_samples).sum(dim=-1) # (n_samples, batch)
        z_samples = z_samples.transpose(0, 1)  # (batch, n_samples, latent_dim)
        log_prob_z_given_x = log_prob_z_given_x.transpose(0, 1)  # (batch, n_samples)
        log_prob_z = self.prior.log_prob(z_samples).sum(dim=-1)

        p_mu_x, p_std_x = self.imputation_decoder(z_samples)
        law_x_given_z = dist.Normal(loc=p_mu_x, scale=p_std_x)
        x_samples = law_x_given_z.rsample() # (batch, n_samples, input_dim)
        log_prob_x_given_z = (law_x_given_z.log_prob(x.unsqueeze(1))*s.unsqueeze(1)).sum(dim=-1) # (batch, n_samples)

        if only_imputation and return_x_samples:
            return q_mu_z, q_log_std_z, p_mu_x, p_std_x, log_prob_z_given_x, log_prob_z, log_prob_x_given_z, x_samples


        # Mask process
        #print(f"S shape: {s.shape}")
        #print(f"X shape: {x.shape}")
        #print(f"X samples shape: {(x_samples.mean(dim=1)).shape}")
        #print((x*s.unsqueeze(1)).shape, (x_samples.mean(dim=1)*(1 - s).unsqueeze(1)).shape)
        mixed_x = (x*s).unsqueeze(1) + x_samples*(1 - s).unsqueeze(1)  # Combine known and imputed values
        #print(mixed_x.shape)
        q_mu_eta, q_log_std_eta = self.encoder_mask(mixed_x)
        law_eta_given_x = dist.Normal(loc=q_mu_eta, scale=torch.exp(0.5 * q_log_std_eta))
        eta_samples = law_eta_given_x.rsample((n_samples,)) # (n_samples, batch, n_samples, latent_dim)
        log_prob_eta_given_x = law_eta_given_x.log_prob(eta_samples).sum(dim=-1)
        eta_samples = eta_samples.transpose(0, 1)           # (batch, n_samples, n_samples, latent_dim)
        log_prob_eta_given_x = log_prob_eta_given_x.transpose(0, 1) # (batch, n_samples, n_samples)
        log_prob_eta = self.prior.log_prob(eta_samples).sum(dim=-1)

        logits_s = self.mask_decoder(eta_samples)
        #print(logits_s.shape)
        #print(s.shape)
        s_expanded = s.unsqueeze(1).expand(-1, n_samples, -1).unsqueeze(1).expand(-1, n_samples, -1, -1)
        law_s_given_eta = dist.Bernoulli(logits=logits_s)
        #print(logits_s.shape, s_expanded.shape)
        log_prob_s_given_eta = law_s_given_eta.log_prob(s_expanded).sum(dim=-1)
        

        # Expand log probabilities for z and x
        log_prob_z_given_x = log_prob_z_given_x.unsqueeze(-1).expand(-1, -1, n_samples)
        log_prob_z = log_prob_z.unsqueeze(-1).expand(-1, -1, n_samples)
        log_prob_x_given_z = log_prob_x_given_z.unsqueeze(-1).expand(-1, -1, n_samples)

        # Return results
        if return_x_samples:
            return q_mu_z, q_log_std_z, p_mu_x, p_std_x, log_prob_z_given_x, log_prob_z, log_prob_x_given_z, q_mu_eta, q_log_std_eta, log_prob_eta_given_x, log_prob_eta, log_prob_s_given_eta, x_samples

        return q_mu_z, q_log_std_z, p_mu_x, p_std_x, log_prob_z_given_x, log_prob_z, log_prob_x_given_z, q_mu_eta, q_log_std_eta, log_prob_eta_given_x, log_prob_eta, log_prob_s_given_eta


def not_miwae_loss_2VAE(log_prob_z_given_x, log_prob_z, log_prob_x_given_z, log_prob_eta_given_x, log_prob_eta, log_prob_s_given_eta):
    elbo = log_prob_z + log_prob_x_given_z + log_prob_s_given_eta - log_prob_z_given_x + log_prob_eta - log_prob_eta_given_x
    # print(elbo.shape) (batch, n_samples)
    return -(torch.mean(torch.logsumexp(elbo, dim=1) - torch.log(torch.tensor([elbo.shape[1]]))))

from sklearn.cluster import KMeans
def kmeans_masking(S, n_clusters = 5):
    # Make clusters for the binary vectors of S
    #S = S.cpu().numpy()
    kmeans = KMeans(n_clusters, random_state=0).fit(S)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Get the cluster of each row in S
    cluster_assignments = kmeans.predict(S)

    return cluster_assignments



def load_model(model_name, model):
    # Load the trained model file
    model_loaded =torch.load(model_name, weights_only = True)

    model.load_state_dict(model_loaded)
    return model


def train_2VAE(model, X_train, S_train, X_val, S_val, batch_size=128, num_epochs=100, n_samples=1, learning_rate=1e-3, patience = 50):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    N = X_train.shape[0]
    
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        p = np.random.permutation(N)
        X_train = X_train[p,:]
        S_train = S_train[p,:]
        
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            s_batch = S_train[i:i+batch_size]

            optimizer.zero_grad()
            q_mu, q_log_std2, p_mu, p_std, log_prob_z_given_x, log_prob_z, log_prob_x_given_z, q_mu_eta, q_log_std2_eta, log_prob_eta_given_x, log_prob_eta, log_prob_s_given_eta = model(x_batch, s_batch, n_samples)
            
            # Calculate loss
            loss = not_miwae_loss_2VAE(log_prob_z_given_x, log_prob_z, log_prob_x_given_z, log_prob_eta_given_x, log_prob_eta, log_prob_s_given_eta) 
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(X_train)
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i in range(0, len(X_val), batch_size):
                x_batch_val = X_val[i:i+batch_size]
                s_batch_val = S_val[i:i+batch_size]
                q_mu, q_log_std2, p_mu, p_std, log_prob_z_given_x, log_prob_z, log_prob_x_given_z, q_mu_eta, q_log_std2_eta, log_prob_eta_given_x, log_prob_eta, log_prob_s_given_eta = model(x_batch_val, s_batch_val, n_samples)
                val_loss += not_miwae_loss_2VAE(log_prob_z_given_x, log_prob_z, log_prob_x_given_z, log_prob_eta_given_x, log_prob_eta, log_prob_s_given_eta).item() * x_batch_val.size(0)
            val_loss /= len(X_val)
            val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    return train_loss_history, val_loss_history

class MissingProcess(nn.Module):
    def __init__(self, input_size, missing='selfmask', activation='tanh', hidden =100):
        super().__init__()
        self.missing = missing
        if self.missing == 'selfmask':
          w_value = torch.randn(size=(input_size,)) / np.sqrt(input_size)
          b_value = torch.randn(size=(input_size,)) / np.sqrt(input_size)
          self.w = nn.Parameter(w_value)
          self.b = nn.Parameter(b_value)
        if self.missing == 'selfmask_known':
          w_value = torch.randn(size=(input_size,)) / np.sqrt(input_size)
          # softplus
          w_value = torch.nn.functional.softplus(w_value)
          self.w = nn.Parameter(w_value)
          b_value = torch.randn(size=(input_size,)) / np.sqrt(input_size)
          self.b = nn.Parameter(b_value)
        if self.missing == 'linear':
          self.w = nn.Linear(input_size, input_size)
        if self.missing == 'non_linear':
          self.w = nn.Sequential(nn.Linear(input_size, hidden), nn.Tanh(), nn.Linear(hidden, input_size))


    def forward(self, x):
        if self.missing == 'selfmask':
          return self.w * x + self.b
        if self.missing == 'selfmask_known':
          # softplus in training
          if self.training:
            self.w = nn.Parameter(nn.functional.softplus(self.w))
          return self.w * x + self.b
        if self.missing == 'linear':
          return self.w(x)
        if self.missing == 'non_linear':
          return self.w(x)

def rmse_imputation(x_orginal, x, s, model, nb_samples = 1_000, device = "cpu"):
    """
    Return the rmse on the missing data and x with the missing values
    """

    x = torch.tensor(x, dtype=torch.float32, device=device)
    s = torch.tensor(s, dtype=torch.float32, device=device)
    x_orginal = torch.tensor(x_orginal, dtype=torch.float32, device=device)

    x_mixed = torch.zeros_like(x_orginal, device=device)
    N = x_orginal.size(0)
    with torch.no_grad():
        for i in range(N):
            x_batch = x[i,:].unsqueeze(0)
            s_batch = s[i,:].unsqueeze(0)
            _, _, _, _, log_prob_s_given_x, log_prob_x_given_z, log_prob_z, log_prob_z_given_x, x_samples = model(x_batch,s_batch ,return_x_samples=True, n_samples=nb_samples) # 4x(batch, n_samples), (batch, n_samples, input_size)

            aks = torch.softmax(log_prob_s_given_x + log_prob_x_given_z + log_prob_z - log_prob_z_given_x, dim = 1) # (batch,n_samples)

            xm = torch.sum(aks.unsqueeze(-1)* x_samples, dim = 1)
            
            x_mixed[i,:] = x_batch * s_batch + (1-s_batch) * xm

        rmse = torch.sqrt(torch.sum(((x_orginal - x_mixed) * (1 - s))**2) / torch.sum(1 - s))

            # rmse2 = torch.sqrt(torch.sum(((x_orginal - x_mixed) **2 * (1 - s))) / torch.sum(1 - s))
            # print( f'{rmse} =? {rmse2}')
        return rmse, x_mixed

def rmse_imputation_2VAE(x_orginal, x, s, model, batch_size = 128, nb_samples = 1_000):

    """
    Return the rmse on the missing data and x with the missing values 
    """

    x = torch.FloatTensor(x)
    s = torch.FloatTensor(s)
    x_orginal = torch.FloatTensor(x_orginal)

    x_mixed = np.zeros_like(x_orginal)#.unsqueeze(1).expand(-1, nb_samples, -1))
    N = x_orginal.size(0)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    #model = model.to(device)
    with torch.no_grad():
        for i in range(N):
            x_batch = x[i:i+batch_size]#.to(device)
            s_batch = s[i:i+batch_size]#.to(device)
            
            _, _, _, _, log_prob_z_given_x, log_prob_z, log_prob_x_given_z, x_samples = model(x_batch,s_batch ,return_x_samples=True, only_imputation = True, n_samples=nb_samples) # 4x(batch, n_samples), (batch, n_samples, input_size)

            # Remove the extra dim in logs
            #print(log_prob_z_given_x.shape, log_prob_z.shape, log_prob_x_given_z.shape)
            #log_prob_z_given_x = log_prob_z_given_x[..., 0]
            #log_prob_z = log_prob_z[..., 0]
            #log_prob_x_given_z = log_prob_x_given_z[..., 0]
            #print(log_prob_z_given_x.shape, log_prob_z.shape, log_prob_x_given_z.shape)

            aks = torch.softmax(log_prob_z + log_prob_x_given_z - log_prob_z_given_x, dim = 1) # (batch,n_samples)
            xm = torch.sum(aks.unsqueeze(-1)* x_samples, dim = 1)

            x_mixed[i:i+batch_size,:] = x_batch.cpu() * s_batch.cpu() + (1-s_batch.cpu()) * xm.cpu()
            torch.cuda.empty_cache()
        
        rmse = torch.sqrt(torch.sum(((x_orginal - x_mixed) * (1 - s))**2) / torch.sum(1 - s))
    #model = model.to("cpu")
    return rmse, x_mixed

