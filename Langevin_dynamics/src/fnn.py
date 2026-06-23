import numpy as np
import torch
import torch.nn as nn

class SimpleFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bi_phfreq_in_thz, freq_in_thz):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self._init_weights(bi_phfreq_in_thz, freq_in_thz)

    def _init_weights(self, bi_phfreq_in_thz, freq_in_thz):
        with torch.no_grad():
            w = torch.tensor([
                [2*np.pi*(1+(bi_phfreq_in_thz/2))/freq_in_thz],
                [2*np.pi*(1-(bi_phfreq_in_thz/2))/freq_in_thz],
                [2*np.pi*(2+(bi_phfreq_in_thz/2))/freq_in_thz],
                [2*np.pi*(2-(bi_phfreq_in_thz/2))/freq_in_thz],
                [2*np.pi*(1.5+(bi_phfreq_in_thz/2))/freq_in_thz],
                [2*np.pi*(1.5-(bi_phfreq_in_thz/2))/freq_in_thz],
                [2*np.pi*(0.5+(bi_phfreq_in_thz/2))/freq_in_thz],
                [2*np.pi*(0.5-(bi_phfreq_in_thz/2))/freq_in_thz],
                [2*np.pi*(2.5+(bi_phfreq_in_thz/2))/freq_in_thz],
                [2*np.pi*(2.5-(bi_phfreq_in_thz/2))/freq_in_thz],
            ], dtype=torch.float32)
            self.fc1.weight = nn.Parameter(w)

        nn.init.normal_(self.fc1.bias, mean=0, std=1)
        nn.init.normal_(self.fc2.weight, mean=0.8, std=1)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        return self.fc2(torch.sin(self.fc1(x)))

    def get_output_batch(self, t_array, device=None):
        if device is None:
            device = next(self.parameters()).device
        t = torch.as_tensor(t_array, dtype=torch.float32, device=device).view(-1, 1)
        with torch.no_grad():
            y = self(t).view(-1)
        return y.detach().cpu().numpy()

def mutate(network, mutation_strength=0.2):
    with torch.no_grad():
        for name, param in network.named_parameters():
            if "fc2.bias" in name:
                continue
            param.add_(torch.randn_like(param) * mutation_strength)

def init_pop(n_pop, input_size, hidden_size, output_size, bi_phfreq_in_thz, freq_in_thz, seed=123):
    torch.manual_seed(seed)
    return [SimpleFNN(input_size, hidden_size, output_size, bi_phfreq_in_thz, freq_in_thz) for _ in range(n_pop)]
