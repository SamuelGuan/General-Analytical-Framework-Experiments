from util.TransformerDecoderOnly import TransformerDecoderOnly
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, Normal, MixtureSameFamily
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import norm

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

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

synthetic_distribution_list = ['normal',
							   'left_skew_distribution',
							   'right_peek_double_skew_distribution',
							   'left_peek_double_skew_distribution']

if __name__ == '__main__':
	'''
	== [Warning! Warning! Warning! Warning! Warning!] ==
	Before running this code, please modify two filepath:
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	->The first is in the file 'component/biSoftmax.py':
	(origin)
	class biSoftmax():
		def __init__():
			...
		->  self.logger_path = f'experiment_data/_[{dataset_name}]_bisoftmax_rank_logger.txt'
			...
	(after modify)
	class biSoftmax():
		def __init__():
			...
		-> self.logger_path = f'../experiment_data/_[{dataset_name}]_bisoftmax_rank_logger.txt'
			..
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	->The second is in the file 'component/FFN.py':
	(origin)
	class FFN():
		def __init__():
			...
		->  self.logger_path = f'experiment_data/_[{dataset_name}]_FFN_rank_logger.txt'
			...
	(after modify)
	class biSoftmax():
		def __init__():
			...
		->  self.logger_path = f'../experiment_data/_[{dataset_name}]_FFN_rank_logger.txt'
			...
	+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	-> or you would meet the bug for the problem 'file can not found!'
	-> After running this file, please modify these two .py back to their origin code!To ensure you
	-> can do other experiments!!!!
	'''
	shape = torch.zeros(size=(512,6)).shape
	o_shape = torch.zeros(size=(128,6)).shape
	for d in tqdm(synthetic_distribution_list):
		output = []
		loss_list =[]
		for i in range(500):
			if d == 'normal':
				input_x = torch.randn(512,6).to('cuda')
				target_y = torch.randn(128,6).to('cuda')
			elif d == 'left_skew_distribution':
				lognormal_dist = torch.distributions.LogNormal(loc=torch.tensor(2.0), scale=torch.tensor(0.3))
				input_x = lognormal_dist.sample(shape).to('cuda')

				lognormal_dist1 = torch.distributions.LogNormal(loc=torch.tensor(1.8), scale=torch.tensor(0.25))
				target_y = lognormal_dist1.sample(o_shape).to('cuda')
			elif d == 'right_peek_double_skew_distribution':
				mix = Categorical(probs=torch.tensor([0.7, 0.3]))
				comp = Normal(
					loc=torch.tensor([-2.5, 1.0]),
					scale=torch.tensor([2.5, 4.0])
				)
				gmm = MixtureSameFamily(mix, comp)
				input_x = gmm.sample(shape).to('cuda')

				mix1 = Categorical(probs=torch.tensor([0.7, 0.3]))
				comp1 = Normal(
					loc=torch.tensor([-2.1, 1.2]),
					scale=torch.tensor([2.8, 4.4])
				)
				gmm1 = MixtureSameFamily(mix1, comp1)
				target_y = gmm1.sample(o_shape).to('cuda')
			else:
				mix = Categorical(probs=torch.tensor([0.3, 0.7]))
				comp = Normal(
					loc=torch.tensor([-2.1, 4.4]),
					scale=torch.tensor([2.8, 1.4])
				)
				gmm = MixtureSameFamily(mix, comp)
				input_x = gmm.sample(shape).to('cuda')

				mix1 = Categorical(probs=torch.tensor([0.3, 0.7]))
				comp1 = Normal(
					loc=torch.tensor([-2.1, 4.4]),
					scale=torch.tensor([2.8, 1.4])
				)
				gmm1 = MixtureSameFamily(mix1, comp1)
				target_y = gmm1.sample(o_shape).to('cuda')

			model = TransformerDecoderOnly(
				sequence_len=512, d_model=6,
				n_head=4, device='cuda', dataset_name=f'synthetic_{d}'
			).to('cuda')

			loss = nn.MSELoss(reduction='mean')

			o = model.forward(x=input_x,
					   		epoch=0,
					   		epochs=1,
					   		mode='eval',
					   		current_batch=0,
					   		total_batch=1)
			l = loss(o, target_y)

			output.append(o.reshape(-1).detach().cpu().numpy())
			loss_list.append(l.reshape(-1).detach().cpu().numpy())

		fig, ax = plt.subplots(figsize=(10, 10),dpi=300)
		loss_list = np.asarray(loss_list).reshape(-1)
		actual_mean = np.mean(loss_list)
		actual_std = np.std(loss_list)
		n, bins, patches = ax.hist(loss_list, density=True, bins=128, color='#8073ac', label='loss density')
		xmin, xmax = plt.xlim()
		x = np.linspace(xmin, xmax, 100)
		p = norm.pdf(x, actual_mean, actual_std)
		ax.plot(x, p, linewidth=3, color='#e08214', label=f'Theoretical Normal')
		ax.grid(True, linestyle='--', alpha=0.7)
		ax.set_title('Loss Distribution', fontsize=36, fontweight='bold', fontname='Times New Roman')
		ax.set_ylabel("Count", fontsize=32, fontweight='bold', fontname='Times New Roman')
		ax.legend(loc='upper left', fontsize=24)
		ax.tick_params(axis='x', labelrotation=0, labelsize=24)
		plt.savefig(f'numeric_figures/_[loss]_synthetic_{d}.pdf')
		plt.close()

