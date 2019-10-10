# Modified from https://github.com/soobinseo/Attentive-Neural-Process/blob/master/network.py
from torch.distributions.normal import Normal
from module import *

class LatentModel(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """
    def __init__(self, num_hidden, x_dim=2, y_dim=1, boolean_pred=True):
        super(LatentModel, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.boolean_pred = boolean_pred
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden,
                input_dim=self.x_dim+self.y_dim)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden,
                x_dim=self.x_dim, y_dim=self.y_dim)
        self.decoder = Decoder(num_hidden,self.x_dim,self.y_dim,self.boolean_pred)
        self.BCELoss = nn.BCELoss()

    def forward(self, context_x, context_y, target_x, target_y=None, test=False):
        num_targets = target_x.size(1)

        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)

        # For training
        if target_y is not None and not test:
            posterior_mu, posterior_var, posterior = self.latent_encoder(
                    target_x, target_y)
            z = posterior

        # For Generation
        else:
            z = prior

        z = z.unsqueeze(1).repeat(1,num_targets,1) # [B, T_target, H]
        r = self.deterministic_encoder(context_x, context_y, target_x) # [B, T_target, H]

        # mu should be the prediction of target y
        mu_pred, log_std = self.decoder(r, z, target_x)

        # For Training
        if target_y is not None:
            if self.boolean_pred:
                log_p = -self.BCELoss(t.sigmoid(mu_pred), target_y)
                mse = None
            else:
                log_p = Normal(loc=mu_pred, scale=t.exp(log_std)).log_prob(target_y)
                # print(log_p.shape)
                log_p = t.mean(log_p)
                mse = t.mean((mu_pred-target_y)**2)
            # get KL divergence between prior and posterior
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
            # maximize prob and minimize KL divergence
            loss = -log_p + kl

        # For Generation
        else:
            log_p = None
            kl = None
            loss = None

        return mu_pred, kl, loss, mse, log_p

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.mean() #.sum()
        return kl_div
