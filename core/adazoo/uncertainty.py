from copy import deepcopy
import os
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from core.param import load_model_and_optimizer, copy_model_and_optimizer

def init_random(bs, *, im_sz, n_ch):
    return torch.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


class UncertaintyModel(nn.Module):
    def __init__(self, model, temperature=0.3, min_temperature=0.05, uncertainty_threshold=0.7, contrast_boost=2.0, noise_boost=1.5):
        super(UncertaintyModel, self).__init__()
        self.f = model
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.initial_temperature = temperature
        self.uncertainty_threshold = uncertainty_threshold
        self.contrast_boost = contrast_boost
        self.noise_boost = noise_boost

    def classify(self, x):
        penult_z = self.f(x)
        return penult_z
    
    def forward(self, x, y=None):
        logits = self.classify(x)
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        uncertainty = entropy / torch.log(torch.tensor(logits.size(1), dtype=torch.float))
        adaptive_temp = self.temperature * (1 + 2 * uncertainty)
        adaptive_temp = adaptive_temp.view(-1, 1)
        if y is None:
            scaled_logits = logits / adaptive_temp
            return scaled_logits.logsumexp(1), logits, uncertainty
        else:
            scaled_logits = logits / adaptive_temp
            return torch.gather(scaled_logits, 1, y[:, None]), logits, uncertainty

    def anneal_temperature(self, epoch, total_epochs):
        """Gradually decrease temperature during training"""
        self.temperature = max(
            self.min_temperature,
            self.initial_temperature * (1 - epoch / total_epochs)
        )


def sample_p_0(reinit_freq, replay_buffer, bs, im_sz, n_ch, device, y=None):
    random_samples = init_random(bs, im_sz=im_sz, n_ch=n_ch)

    if len(replay_buffer) == 0:
        return random_samples.to(device), []
    buffer_size = len(replay_buffer)
    inds = torch.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds

    buffer_samples = replay_buffer[inds]

    if buffer_samples.shape != random_samples.shape:
        print(
            f"[WARNING] Buffer shape {buffer_samples.shape} doesn't match random shape {random_samples.shape}. Discarding buffer.")
        return random_samples.to(device), inds

    choose_random = (torch.rand(bs) < reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples

    return samples.to(device), inds


def sample_q(f, replay_buffer, n_steps, sgld_lr, sgld_std, reinit_freq, batch_size, im_sz, n_ch, device, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()

    # get batch size
    bs = batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(reinit_freq=reinit_freq, replay_buffer=replay_buffer, bs=bs, im_sz=im_sz, n_ch=n_ch, device=device ,y=y)
    init_samples = deepcopy(init_sample)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(n_steps):
        f_prime = torch.autograd.grad(f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    if final_samples.shape[1:] != replay_buffer.shape[1:]:
        raise RuntimeError(f"Shape mismatch: sample {final_samples.shape} vs buffer {replay_buffer.shape}")
    assert final_samples.shape[1:] == replay_buffer.shape[1:], \
        f"Replay buffer shape {replay_buffer.shape[1:]} doesn't match generated sample shape {final_samples.shape[1:]}"

    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples, init_samples.detach()

class Uncertainty(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, 
                 buffer_size=10000, sgld_steps=20, sgld_lr=1, sgld_std=0.01, reinit_freq=0.05, if_cond=False, 
                 n_classes=10, im_sz=64, n_ch=3, path=None, logger=None,
                 temperature=0.3, min_temperature=0.05, uncertainty_threshold=0.7,
                 contrast_boost=2.0, noise_boost=1.5):
        super().__init__()

        self.im_sz = im_sz
        self.n_ch = n_ch
        self.uncertainty_model = UncertaintyModel(model, 
                                                temperature=temperature,
                                                min_temperature=min_temperature,
                                                uncertainty_threshold=uncertainty_threshold,
                                                contrast_boost=contrast_boost,
                                                noise_boost=noise_boost)
        self.replay_buffer = init_random(buffer_size, im_sz=self.im_sz, n_ch=self.n_ch)

        self.replay_buffer_old = deepcopy(self.replay_buffer)
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.sgld_steps = sgld_steps
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self.reinit_freq = reinit_freq
        self.if_cond = if_cond
        
        self.n_classes = n_classes
        self.path=path
        self.logger = logger

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.uncertainty_model, self.optimizer)

    def forward(self, x, if_adapt=True, counter=None, if_vis=False):
        if self.episodic:
            self.reset()
        
        if if_adapt:
            for i in range(self.steps):
                outputs = forward_and_adapt(x, self.uncertainty_model, self.optimizer, 
                                            self.replay_buffer, self.sgld_steps, self.sgld_lr, self.sgld_std, self.reinit_freq,
                                            if_cond=self.if_cond, n_classes=self.n_classes)
                if i % 1 == 0 and if_vis:
                    visualize_images(path=self.path, replay_buffer_old=self.replay_buffer_old, replay_buffer=self.replay_buffer, uncertainty_model=self.uncertainty_model, 
                                    sgld_steps=self.sgld_steps, sgld_lr=self.sgld_lr, sgld_std=self.sgld_std, reinit_freq=self.reinit_freq,
                                    batch_size=100, n_classes=self.n_classes, im_sz=self.im_sz, n_ch=self.n_ch, device=x.device, counter=counter, step=i)
        else:
            self.uncertainty_model.eval()
            with torch.no_grad():
                outputs = self.uncertainty_model.classify(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.uncertainty_model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
@torch.enable_grad()
def visualize_images(path, replay_buffer_old, replay_buffer, uncertainty_model, 
                     sgld_steps, sgld_lr, sgld_std, reinit_freq,
                     batch_size, n_classes, im_sz, n_ch, device=None, counter=None, step=None):
    num_cols=10
    repeat_times = batch_size // n_classes
    y = torch.arange(n_classes).repeat(repeat_times).to(device) 
    x_fake, _ = sample_q(uncertainty_model, replay_buffer, n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=y)
    images = x_fake.detach().cpu()
    save_image(images , os.path.join(path, 'sample.png'), padding=2, nrow=num_cols)

    num_cols=40
    images_init = replay_buffer_old.cpu()
    images = replay_buffer.cpu() 
    images_diff = replay_buffer.cpu() - replay_buffer_old.cpu()
    if step == 0:
        save_image(images_init , os.path.join(path, 'buffer_init.png'), padding=2, nrow=num_cols)
    save_image(images , os.path.join(path, 'buffer-'+str(counter)+"-"+str(step)+'.png'), padding=2, nrow=num_cols) # 
    save_image(images_diff , os.path.join(path, 'buffer_diff.png'), padding=2, nrow=num_cols)

@torch.enable_grad()
def forward_and_adapt(x, uncertainty_model, optimizer, replay_buffer, sgld_steps, sgld_lr, sgld_std, reinit_freq, if_cond=False, n_classes=10):
    batch_size = x.shape[0]
    n_ch = x.shape[1]
    im_sz = x.shape[2]
    device = x.device

    if if_cond == 'uncond':
        x_fake, _ = sample_q(uncertainty_model, replay_buffer, 
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, 
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=None)
    elif if_cond == 'cond':
        y = torch.randint(0, n_classes, (batch_size,)).to(device)
        x_fake, _ = sample_q(uncertainty_model, replay_buffer, 
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, 
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=y)

    # forward
    out_real = uncertainty_model(x)
    uncertainty_real = out_real[0]
    uncertainty_real_full = out_real[2]
    logits_real = out_real[1]
    
    out_fake = uncertainty_model(x_fake)
    uncertainty_fake = out_fake[0]
    uncertainty_fake_full = out_fake[2]
    logits_fake = out_fake[1]

    # Compute energy terms
    energy_real = -logits_real.logsumexp(1).mean()
    energy_fake = -logits_fake.logsumexp(1).mean()
    energy_loss = -(energy_real - energy_fake)

    # Compute uncertainty terms
    mean_uncertainty_real = uncertainty_real.mean()
    mean_uncertainty_fake = uncertainty_fake.mean()
    var_uncertainty_real = uncertainty_real.var()
    var_uncertainty_fake = uncertainty_fake.var()

    # Enhanced uncertainty handling
    uncertainty_diff = mean_uncertainty_real - mean_uncertainty_fake
    uncertainty_factor = torch.sigmoid(uncertainty_real_full.mean() - uncertainty_model.uncertainty_threshold)
    variance_weight = 0.1 * (1 + torch.sigmoid(uncertainty_diff)) * (1 + 2 * uncertainty_factor)
    contrast_aware_term = torch.mean(torch.abs(uncertainty_real_full - uncertainty_fake_full)) * uncertainty_model.contrast_boost
    noise_aware_term = torch.mean(torch.abs(uncertainty_real - uncertainty_fake)) * uncertainty_model.noise_boost

    # Combine both losses
    uncertainty_loss = (- (mean_uncertainty_real - mean_uncertainty_fake) + 
                       variance_weight * (var_uncertainty_real + var_uncertainty_fake) +
                       contrast_aware_term +
                       noise_aware_term)

    # Combine both losses with a weighting factor
    energy_weight = 0.5
    uncertainty_weight = 0.5
    loss = energy_weight * energy_loss + uncertainty_weight * uncertainty_loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    outputs = uncertainty_model.classify(x)

    return outputs
