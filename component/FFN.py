import torch
import torch.nn as nn
import time

class FFN(nn.Module):
	def __init__(self,
				 sequence_len:int,
				 dataset_name: str,
				 device='cuda'):
		super(FFN, self).__init__()
		self.sequence_len = sequence_len
		self.device = device
		# If you want to run the output_distribution_stimulation.py, change this:
		self.logger_path = f'experiment_data/_[{dataset_name}]_FFN_rank_logger.txt'
		####################################################################################
		self.ffn = nn.Linear(in_features=sequence_len, out_features=sequence_len, bias = False)
		self.relu = nn.ReLU()
		self.Wo = nn.Linear(in_features=sequence_len, out_features=sequence_len, bias = False)

		nn.init.xavier_normal_(self.Wo.weight, gain=nn.init.calculate_gain('linear'))
		nn.init.xavier_normal_(self.ffn.weight, gain=nn.init.calculate_gain('relu'))

		# cleaning the logger for every new model training
		with open(self.logger_path, 'w') as f:
			f.write(f'Train_start_time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
			f.write(f'''
=======================================================
FFN config:
[ffn]shape: [{self.ffn.weight.shape}], bias: [False]
[Wo]shape: [{self.Wo.weight.shape}], bias: [False]
=======================================================
''')

	def forward(self, x:torch.Tensor, epoch:int, mode:str, current_batch:int, total_batch:int) -> torch.Tensor:
		if len(x.size()) != 3 and x.size()[-1] != self.sequence_len:
			raise ValueError(f'x must have three dimensions and last dimension must be sequence_len! But get {x.size()}!')
		norm_x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-9)
		ffn_x = (self.relu(self.ffn(norm_x.transpose(-1,-2)))/self.sequence_len).transpose(-1,-2)
		add_x = ffn_x + x
		sum_x = torch.sum(add_x, dim=0)
		output = self.Wo(sum_x)
		if mode=='train' and current_batch == total_batch-1:
			self.rank_cal(epoch)
		return output

	# calculate the rank of linear when training
	def rank_cal(self, epoch) -> None:
		ffn_weight = self.ffn.weight
		Wo_weight = self.Wo.weight
		logger_info1 = f'epoch: {epoch}, FFN layer rank: {torch.linalg.matrix_rank(ffn_weight)}\n'
		logger_info2 = f'epoch: {epoch}, Output layer rank: {torch.linalg.matrix_rank(Wo_weight)}\n'
		with open(self.logger_path, 'a') as f:
			f.write(logger_info1)
			f.write(logger_info2)

