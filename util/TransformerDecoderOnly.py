import torch
import torch.nn as nn
from typing import Optional
from component.biSoftmax import biSoftmax
from component.FFN import FFN
from component.PositionEmbedding import PositionalEncoding

class TransformerDecoderOnly(nn.Module):
	def __init__(self,
				 sequence_len: int,
				 d_model: int,
				 n_head: int,
				 dataset_name:str,
				 device='cuda'
				 ):
		super(TransformerDecoderOnly, self).__init__()
		self.device = device
		self.sequence_len = sequence_len
		self.subsequence_len = sequence_len // n_head
		if sequence_len % n_head != 0:
			raise ValueError('sequence length must be int-divisible by n_head!')
		self.d_model = d_model
		self.n_head = n_head
		self.biSoftmax = biSoftmax(
			n_head=n_head,
			d_model=self.subsequence_len,
			sequence_len=self.subsequence_len,
			device=device,
			dataset_name=dataset_name,
		)
		self.FFN = FFN(
			sequence_len=self.subsequence_len,
			device=device,
			dataset_name=dataset_name,
		)
		self.PositionalEncoding = PositionalEncoding(
			sequence_len=sequence_len,
			device=device
		)
		self.take = nn.Linear(d_model, self.subsequence_len, bias=False)
		self.retake = nn.Linear(self.subsequence_len, d_model, bias=False)
		nn.init.xavier_normal_(self.take.weight)
		nn.init.xavier_normal_(self.retake.weight)

	def forward(self, x: torch.Tensor, epoch:int, epochs:int, mode:str, current_batch:int, total_batch:int)-> torch.Tensor:
		if len(x.size()) != 2:
			raise ValueError(f"x must 2d tensor!got{x.size()}")
		pe_x = self.PositionalEncoding(x.transpose(-1,-2)).transpose(-1,-2)
		mean = torch.mean(pe_x,dim=-2,keepdim=True)
		std = torch.std(pe_x,dim=-2,keepdim=True)
		norm_x = (pe_x-mean) / std
		norm_x1 = self.take(norm_x)
		if mode=='train':
			bisoftmax_x = self.biSoftmax.forward(x=norm_x1, epoch=epoch, epochs=epochs, mode=mode,
												 current_batch=current_batch, total_batch=total_batch)
			ffn_x = self.FFN.forward(x=bisoftmax_x, epoch=epoch, mode=mode,
									current_batch=current_batch, total_batch=total_batch)
		else:
			bisoftmax_x = self.biSoftmax.forward(x=norm_x1,epoch=epoch, epochs=epochs, mode='eval',
												 current_batch=current_batch, total_batch=total_batch)
			ffn_x = self.FFN.forward(x=bisoftmax_x, epoch=epoch, mode='eval',
									current_batch=current_batch, total_batch=total_batch)
		denormalize_ffn_x = self.retake(ffn_x) * std + mean
		return denormalize_ffn_x
