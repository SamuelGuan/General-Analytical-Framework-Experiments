import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, sequence_len, device='cuda'):
        super().__init__()
        self.device = device
        m = torch.Tensor(torch.arange(0, sequence_len, 2)).to(self.device)
        if sequence_len % 2 != 0:
            raise ValueError(f'd_model should be even! Got {sequence_len}!')
        theta = 10000 ** (-2 * torch.arange(0, sequence_len, 2, device=device)/sequence_len)
        cos_theta = torch.cos(m*theta)
        sin_theta = torch.sin(m*theta)
        neg_sin_theta = -sin_theta
        part1 = torch.zeros(size=[sequence_len]).to(self.device)
        part1[0::2] = cos_theta
        part1[1::2] = sin_theta
        self.part1 = part1

        part2 = torch.zeros(size=[sequence_len]).to(self.device)
        part2[0::2] = neg_sin_theta
        part2[1::2] = cos_theta
        self.part2 = part2




    def forward(self, x:torch.Tensor)->torch.Tensor:
        with torch.no_grad():
            half_first_term = torch.zeros_like(x).to(self.device)
            half_first_term[0::2] = x[0::2]
            half_first_term[1::2] = x[0::2]
            first_term = self.part1 * half_first_term

            half_second_term = torch.zeros_like(x).to(self.device)
            half_second_term[0::2] = x[1::2]
            half_second_term[1::2] = x[1::2]
            second_term = self.part2 * half_second_term

        return first_term + second_term