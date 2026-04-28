import torch
import torch.nn.functional as F

def reconstruction_loss(x, x_tilde):
    return F.mse_loss(x, x_tilde)

def supervised_loss(h, h_sup):
    return F.mse_loss(h[:, 1:, :], h_sup[:, :-1, :])

def generator_loss(y_fake):
    return -torch.mean(y_fake)

def discriminator_loss(y_real, y_fake):
    return torch.mean(y_fake) - torch.mean(y_real)

def train_embedder(model, x, optimizer):
    h = model.embedder(x)
    x_tilde = model.recovery(h)

    loss = reconstruction_loss(x, x_tilde)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_supervisor(model, x, optimizer):
    h = model.embedder(x).detach()
    h_sup = model.supervisor(h)

    loss = supervised_loss(h, h_sup)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_joint(model, x, z, optim_G, optim_D):

    # ======================
    # Generator + Supervisor
    # ======================
    h = model.embedder(x)

    h_hat = model.generator(z)
    h_hat_sup = model.supervisor(h_hat)

    y_fake = model.discriminator(h_hat_sup)

    g_loss = generator_loss(y_fake)

    # supervised consistency
    h_sup = model.supervisor(h)
    s_loss = supervised_loss(h, h_sup)

    loss_G = g_loss + 100 * s_loss

    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()

    # ======================
    # Discriminator
    # ======================
    y_real = model.discriminator(h.detach())
    y_fake = model.discriminator(h_hat_sup.detach())

    d_loss = discriminator_loss(y_real, y_fake)

    optim_D.zero_grad()
    d_loss.backward()
    optim_D.step()

    return loss_G.item(), d_loss.item()

def generate(model, batch_size, seq_len, noise_dim, device=None):
    if device is None:
        device = next(model.parameters()).device

    z = torch.randn(batch_size, seq_len, noise_dim, device=device)

    h_hat = model.generator(z)
    h_hat_sup = model.supervisor(h_hat)

    x_hat = model.recovery(h_hat_sup)

    return x_hat