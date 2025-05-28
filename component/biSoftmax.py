import math
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

# utilize Latex
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif",
	"font.serif": ["Computer Modern Roman"],
	"mathtext.fontset": "cm",
})

plt.rcParams['figure.dpi'] = 300
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['axes.linewidth'] = 1.5

class biSoftmax(nn.Module):
	def __init__(self,
				 sequence_len:int,
				 d_model:int,
				 n_head:int,
				 dataset_name:str,
				 device='cuda'):
		super(biSoftmax, self).__init__()
		self.sequence_len = sequence_len
		self.d_model = d_model
		self.n_head = n_head
		self.device = device
		#If you want to run the output_distribution_stimulation.py, change this:
		self.logger_path = f'experiment_data/_[{dataset_name}]_bisoftmax_rank_logger.txt'
		####################################################################################
		self.dataset_name = dataset_name

		self.Wq1 = nn.Linear(sequence_len, sequence_len, bias=False)
		self.Wk1 = nn.Linear(sequence_len, sequence_len, bias=False)
		self.Wq2 = nn.Linear(sequence_len, sequence_len, bias=False)
		self.Wk2 = nn.Linear(sequence_len, sequence_len, bias=False)

		self.softmax1 = nn.Softmax(dim=-1)
		self.softmax2 = nn.Softmax(dim=-1)

		self.W1 = nn.Linear(sequence_len, sequence_len, bias=False)
		self.W2 = nn.Linear(sequence_len, sequence_len, bias=False)

		self.Wv = nn.Linear(sequence_len, sequence_len, bias=False)

		nn.init.xavier_normal_(self.Wq1.weight, gain=nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_normal_(self.Wq2.weight, gain=nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_normal_(self.Wk1.weight, gain=nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_normal_(self.Wk2.weight, gain=nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_normal_(self.Wv.weight, gain=nn.init.calculate_gain('linear'))
		nn.init.xavier_normal_(self.W1.weight, gain=nn.init.calculate_gain('linear'))
		nn.init.xavier_normal_(self.W2.weight, gain=nn.init.calculate_gain('linear'))

		# cleaning the logger for every new model training
		with open(self.logger_path, 'w') as f:
			f.write(f'Train_start_time: {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}\n')
			f.write(f'''
=======================================================
bisoftmax config: 
[Wq1]shape: [{self.Wq1.weight.shape}], bias: [False]
[Wq2]shape: [{self.Wq2.weight.shape}], bias: [False]
[Wk1]shape: [{self.Wk1.weight.shape}], bias: [False]
[Wk2]shape: [{self.Wk2.weight.shape}], bias: [False]
[Wv] shape: [{self.Wv.weight.shape}], bias: [False]
=======================================================
''')


	def forward(self, x:torch.Tensor, epoch:int, epochs:int, mode:str, current_batch:int, total_batch:int) -> torch.Tensor:
		if len(x.size()) != 2 and x.size()[0] != self.n_head * self.subsequence_len:
			raise ValueError(f'x must be 2d tensor and row must be equal to n_head * sequence_len, got {x.size()} instead!')
		reshape_x = x.reshape(self.n_head, self.sequence_len, self.d_model)
		q1 = self.Wq1(reshape_x)
		q2 = self.Wq2(reshape_x)
		k1 = self.Wk1(reshape_x)
		k2 = self.Wk2(reshape_x)
		v = self.Wv(reshape_x)

		score1 = self.W1(self.softmax1(q1 @ k1.transpose(-1, -2) / math.sqrt(self.sequence_len)))
		score2 = self.W2(self.softmax2(q2 @ k2.transpose(-1, -2) / math.sqrt(self.sequence_len)))

		circle_operator = (score1 - score2) * (score1 - score2)

		result = circle_operator @ v

		if mode == "train" and current_batch == total_batch-1:
			self.cal_rank(self.Wq1.weight, 'Wq1', epoch)
			self.cal_rank(self.Wq2.weight, 'Wq2', epoch)
			self.cal_rank(self.Wk1.weight, 'Wk1', epoch)
			self.cal_rank(self.Wk2.weight, 'Wk2', epoch)
			self.cal_rank(self.Wv.weight, 'Wv', epoch)
			self.cal_rank(self.W1.weight, 'W1', epoch)
			self.cal_rank(self.W2.weight, 'W2', epoch)
		return result

	def cal_rank(self, tensor:torch.Tensor, tensor_name:str, epoch:int) -> None:
		logger_info = f'epoch: {epoch}, [{tensor_name}] layer rank: {torch.linalg.matrix_rank(tensor)}\n'
		with open(self.logger_path, 'a') as f:
			f.write(logger_info)

	def drawHeatMap(self,
					attention_matrix:torch.Tensor,
					title:str,
					figsize=(10, 8),
					dpi=300) -> None:
		"""
		draw heatmap for each attention scores head
		"""
		colors = ['#542788','#8073ac', '#b2abd2', '#d8daeb', '#fee0b6', '#fdb863', '#e08214', '#b35806']
		cmap1 = LinearSegmentedColormap.from_list("BlueRed", colors)

		if isinstance(attention_matrix, torch.Tensor):
			attention_matrix_np = attention_matrix.detach().cpu().numpy()
		else:
			attention_matrix_np = attention_matrix

		for k in range(self.n_head):
			new_title = f'{title}.head{k} Heatmap'
			plt.figure(figsize=figsize, dpi=dpi)
			matrix = attention_matrix_np[k]
			scale = np.max(matrix, keepdims=False) - np.min(matrix, keepdims=False)
			lower_bound = np.min(matrix, keepdims=False)
			attn_score = (matrix - lower_bound) / scale

			im = plt.imshow(attn_score, cmap=cmap1, interpolation='nearest')

			cbar = plt.colorbar(im)
			cbar.set_label('Attention weight', fontname='Times New Roman', fontsize=24)

			plt.title(new_title, fontname='Times New Roman', fontweight='bold', fontsize=24)

			plt.tight_layout()
			plt.savefig(f'experiment_result_figure/_[{self.dataset_name}]_{new_title}.pdf')
			plt.show()
