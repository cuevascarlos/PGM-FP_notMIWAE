import matplotlib.pyplot as plt
import torch
import numpy as np
import os


def plot_function_history(train_history, val_history, title=None):
    epochs = range(1, len(train_history) + 1)
    
    plt.plot(epochs, train_history, 'b', label='Training')
    plt.plot(epochs, val_history, 'r', label='Validation')
    main_title = title.rsplit('/',1)[-1]
    plt.title(main_title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Save the plot
    if title is not None:
        plt.savefig(f'{title}.png')
    else:
        plt.savefig('loss_history.png')
    
    #plt.show()
    plt.close()

def ploting_latent(x, s, model, y, nb_samples = 1_000, components = 2, title = None):

    #x = torch.FloatTensor(x)
    #s = torch.FloatTensor(s)
    #y = torch.FloatTensor(y)
    N = x.size(0)

    latents_mu_eta = []
    latents_std_eta = []
    latents_mu_z = []
    latents_std_z = []
    labels = []

    with torch.no_grad():
        for i in range(N):
            x_batch = x[i,:].unsqueeze(0)
            s_batch = s[i,:].unsqueeze(0)
            label = y[i]
            q_mu_z, q_log_std_z, p_mu_x, p_std_x, _, _, _, q_mu_eta, q_log_std2_eta, _, _, _, _ = model(x_batch,s_batch ,return_x_samples=True, n_samples=nb_samples) # 4x(batch, n_samples), (batch, n_samples, input_size)

            latents_mu_eta.append(q_mu_eta)
            latents_std_eta.append(q_log_std2_eta)
            latents_mu_z.append(q_mu_z)
            latents_std_z.append(q_log_std_z)
            labels.append(label)

    latents_mu_eta = torch.cat(latents_mu_eta, dim = 0)
    latents_std_eta = torch.cat(latents_std_eta, dim = 0)
    latents_mu_z = torch.cat(latents_mu_z, dim = 0)
    latents_std_z = torch.cat(latents_std_z, dim = 0)
    #labels = torch.cat(labels)

    tsne_plot(latents_mu_z, labels, components = components, comp_plot = components, title = f'{title}-Latent space z')
    tsne_plot(latents_mu_eta, labels, components = components, comp_plot = components, title = f'{title}-Latent space eta')
    

    #plt.show()


from sklearn.manifold import TSNE
def tsne_plot(x, y, components = 2, comp_plot = 2, title = None):
    if x.dim() > 2:
        n_samples = x.size(1)
        x= x.view(-1, x.size(-1))
        y = torch.tensor(y)
        y = y.repeat_interleave(n_samples)
        
    tsne = TSNE(n_components=components, verbose=1, random_state=42)
    tsne_results = tsne.fit_transform(x)
    fig = plt.figure()
    if comp_plot == 2:
        plt.scatter(tsne_results[:,0], tsne_results[:,1], c = y)
    elif comp_plot == 3 and components >= 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], c = y)

    if title is not None:
        main_title = title.rsplit('/',1)[-1] 
        root = title.rsplit('/',1)[0]
        plt.title(main_title)
        plt.savefig(f"{root}/{main_title}_{components}D.png")
    
    if comp_plot == 3:
        plt.show()
    plt.close()
