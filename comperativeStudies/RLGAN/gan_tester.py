import torch
from torch.nn.functional import binary_cross_entropy

torch.manual_seed(0)


"""--------------------------------------------------loss and optim--------------------------------------------------"""
Tensor = torch.FloatTensor

def test_model(test_loader, state_dict, generator, discriminator):
    generator.load_state_dict(torch.load(f"{state_dict}")["gen"])
    discriminator.load_state_dict(torch.load(f"{state_dict}")["disc"])    
    
    generator.eval()
    discriminator.eval()

    total_g_loss = 0.0
    total_d_loss = 0.0

    with torch.no_grad():
        for X, y in test_loader:
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

        avg_g_loss = total_g_loss / len(test_loader)
        avg_d_loss = total_d_loss / len(test_loader)

    return avg_g_loss, avg_d_loss
