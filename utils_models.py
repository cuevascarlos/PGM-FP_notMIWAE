import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist

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

    def forward(self, x, s, n_samples=1, return_x_samples=False):
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
        log_prob_x_given_z = (law_x_given_z.log_prob(x.unsqueeze(1)) * s.unsqueeze(1)).sum(dim=-1)

        # Mask process
        mixed_x = x * s + p_mu_x.mean(dim=1) * (1 - s)  # Combine known and imputed values
        q_mu_eta, q_log_std_eta = self.encoder_mask(mixed_x)
        law_eta_given_x = dist.Normal(loc=q_mu_eta, scale=torch.exp(0.5 * q_log_std_eta))
        eta_samples = law_eta_given_x.rsample((n_samples,)) # (n_samples, batch, latent_dim)
        log_prob_eta_given_x = law_eta_given_x.log_prob(eta_samples).sum(dim=-1)
        eta_samples = eta_samples.transpose(0, 1)           # (batch, n_samples, latent_dim)
        log_prob_eta_given_x = log_prob_eta_given_x.transpose(0, 1) # (batch, n_samples)
        log_prob_eta = self.prior.log_prob(eta_samples).sum(dim=-1)

        logits_s = self.mask_decoder(eta_samples)
        #print(s.shape)
        s_expanded = s.unsqueeze(1).expand(-1, n_samples, -1)
        law_s_given_eta = dist.Bernoulli(logits=logits_s)
        #print(logits_s.shape, s_expanded.shape)
        log_prob_s_given_eta = law_s_given_eta.log_prob(s_expanded).sum(dim=-1)

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

def rmse_imputation_2VAE(x_orginal, x, s, model, nb_samples = 1_000):

    """
    Return the rmse on the missing data and x with the missing values 
    """

    x = torch.FloatTensor(x)
    s = torch.FloatTensor(s)
    x_orginal = torch.FloatTensor(x_orginal)

    x_mixed = np.zeros_like(x_orginal)
    N = x_orginal.size(0)
    with torch.no_grad():
        for i in range(N):
            x_batch = x[i,:].unsqueeze(0)
            s_batch = s[i,:].unsqueeze(0)
            _, _, _, _, log_prob_z_given_x, log_prob_z, log_prob_x_given_z, q_mu_eta, q_log_std2_eta, log_prob_eta_given_x, log_prob_eta, log_prob_s_given_eta, x_samples = model(x_batch,s_batch ,return_x_samples=True, n_samples=nb_samples) # 4x(batch, n_samples), (batch, n_samples, input_size)

            aks = torch.softmax(log_prob_z + log_prob_x_given_z + log_prob_s_given_eta - log_prob_z_given_x + log_prob_eta - log_prob_eta_given_x, dim = 1) # (batch,n_samples)
            
            xm = torch.sum(aks.unsqueeze(-1)* x_samples, dim = 1)

            x_mixed[i,:] = x_batch * s_batch + (1-s_batch) * xm

        rmse = torch.sqrt(torch.sum(((x_orginal - x_mixed) * (1 - s))**2) / torch.sum(1 - s))
        return rmse, x_mixed

