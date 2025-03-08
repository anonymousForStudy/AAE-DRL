"""-----------------------------------------------import libraries-----------------------------------------------"""
from torch.nn.functional import binary_cross_entropy
import torch

torch.manual_seed(0)

"""--------------------------------------------------loss and optim--------------------------------------------------"""

def gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

Tensor = torch.FloatTensor


"""--------------------------------------------------model training--------------------------------------------------"""
def train_model(train_loader, generator, discriminator, optimizer_D, optimizer_G):
    discriminator.train()
    generator.train()

    g_total =0
    d_total = 0

    for i, (X, y) in enumerate(train_loader):
        real_latent = X.type(Tensor)
        real_labels = torch.ones(X.shape[0], 1).type(Tensor)
        fake_labels = torch.zeros(X.shape[0], 1).type(Tensor)

        # ================== Train Discriminator ================== #
        optimizer_D.zero_grad()

        # Real data
        d_out_real, _ = discriminator(real_latent)
        d_loss_real = binary_cross_entropy(d_out_real, real_labels)

        # Fake data
        z = torch.randn(X.shape[0], 10).type(Tensor)
        fake_latent, _ = generator(z)
        d_out_fake, _ = discriminator(fake_latent.detach())  # detach to avoid training G
        d_loss_fake = binary_cross_entropy(d_out_fake, fake_labels)


        # Total discriminator loss
        d_loss = 0.5*(d_loss_real + d_loss_fake)

        alpha = torch.rand(real_latent.size(0), 1)
        interpolates = alpha * real_latent + (1 - alpha) * fake_latent
        interpolates.requires_grad_(True)

        d_interpolates, _ = discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        d_loss = 0.2 * gradient_penalty + d_loss
        d_loss.backward()
        optimizer_D.step()

        # ================== Train Generator ================== #
        optimizer_G.zero_grad()

        # Generate fake data
        z = torch.randn(X.shape[0], 10).type(Tensor)
        fake_latent, _ = generator(z)
        g_out_fake, _ = discriminator(fake_latent)

        # Generator loss
        g_loss = g_out_fake.mean()

        g_loss.backward()
        optimizer_G.step()

        # Keep track of losses
        g_total += g_loss.item()
        d_total += d_loss.item()

    g_total_loss = g_total / len(train_loader)
    d_total_loss = d_total / len(train_loader)

    return g_total_loss, d_total_loss


"""-------------------------------------------------model validation-------------------------------------------------"""

def evaluate_model(val_loader, generator, discriminator):
    generator.eval()
    discriminator.eval()

    total_g_loss = 0.0
    total_d_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            real = X.type(Tensor)
            real_labels = torch.ones(X.shape[0], 1).type(Tensor)
            fake_labels = torch.zeros(X.shape[0], 1).type(Tensor)

            d_out_real, _ = discriminator(real)
            d_loss_real = binary_cross_entropy(d_out_real, real_labels)

            # Fake data
            z = torch.randn(X.shape[0], 10).type(Tensor)
            fake_latent, _ = generator(z)
            d_out_fake, _ = discriminator(fake_latent.detach())  # detach to avoid training G
            d_loss_fake = binary_cross_entropy(d_out_fake, fake_labels)

            # Total discriminator loss
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # apply Gumbel Softmax
            z = torch.randn(X.shape[0], 10).type(Tensor)
            fake_latent, _ = generator(z)
            g_out_fake, _ = discriminator(fake_latent)

            # Generator loss
            g_loss = g_out_fake.mean()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        avg_g_loss = total_g_loss / len(val_loader)
        avg_d_loss = total_d_loss / len(val_loader)


    return avg_g_loss, avg_d_loss




