import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from util.TSdatahanding import DataProcessor
from util.TransformerDecoderOnly import TransformerDecoderOnly
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

class MAIN(object):
	_optimizer_ = ['gd', 'sgd']

	def __init__(self,
				 file_path:str,
				 dataset_name:str,
				 columns:list,
				 optimizer:str,
				 device='cuda'):
		self.device = device
		self.columns = columns
		self.data = DataProcessor(
			file_path=file_path,
			split_rate=0.8,
			columns=columns
		)
		self.last_epoch = 0
		self.dataset_name = dataset_name
		self.optimizer = optimizer
		if optimizer not in self._optimizer_:
			raise ValueError(f"optimzer must be 'gd' or 'sgd'! got {optimizer} instead!")

	def train(self, sequence_len:int, n_head:int, lr_schema:float, mini_batch_size=-1, epochs=1, Continue=False)->list:
		model = TransformerDecoderOnly(
			sequence_len=sequence_len,d_model=len(self.columns),
			n_head=n_head, device=self.device ,dataset_name=self.dataset_name).to(self.device)
		if Continue:
			checkpoint = torch.load(f'model_parameters/_[{self.dataset_name}]_.pth')
			model.load_state_dict(checkpoint)

		if self.optimizer == 'gd':
			optimizer = SGD(model.parameters(),lr=lr_schema)
		else:
			optimizer = SGD(model.parameters(),lr=lr_schema)
		criterion = nn.MSELoss(reduction='sum').to(self.device)

		# register_data so that we didn't need to load dataset from outer to memory
		self.data.register_data(sequence_len=sequence_len, output_len=sequence_len//n_head)
		if mini_batch_size == -1:
			mini_batch_size = self.data.train_dataset_size

		# loss list for ploting
		epoch_mean_train_loss_list = []
		epoch_mean_eval_loss_list = []
		output_list = []

		pbar = tqdm(range(epochs), desc='training')
		for epoch in pbar:
			epoch_train_loss = 0
			epoch_train_count = 0
			# gradient accumulator
			for param in model.parameters():
				param.grad = torch.zeros_like(param.data).to(self.device)

			# use mini-batch
			for inputs, outputs in self.data.get_dataloader(mode='train', mini_batch_size=mini_batch_size):
				train_pred = model.forward(x=inputs.to(self.device),
										   epoch=epoch,
										   epochs=epochs,
										   mode='train',
											current_batch=epoch_train_count,
											total_batch=mini_batch_size)
				if mini_batch_size - epoch_train_count == 1:
					output_list.append(train_pred.cpu().detach().numpy())
				loss = criterion(train_pred, outputs.to(self.device))
				loss.backward()
				epoch_train_loss += loss.item()
				epoch_train_count += 1

			# mean loss
			epoch_mean_train_loss_list.append(epoch_train_loss / epoch_train_count)
			for param in model.parameters():
				if param.grad is not None:
					param.grad /= epoch_train_count

			optimizer.step()
			optimizer.zero_grad()

			epoch_eval_loss = 0
			epoch_eval_count = 0
			model.eval()
			with torch.no_grad():
				for inputs, outputs in self.data.get_dataloader(mode='eval', mini_batch_size=mini_batch_size):
					eval_pred = model.forward(x=inputs.to(self.device),
											  epoch=epoch,
											  epochs=epochs,
											  mode='eval',
											  current_batch=epoch_eval_count,
											  total_batch=mini_batch_size)
					epoch_eval_loss += criterion(eval_pred, outputs.to(self.device)).item()
					epoch_eval_count += 1
			epoch_mean_eval_loss_list.append(epoch_eval_loss / epoch_eval_count)
			model.train()

			# update the training bar
			pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
			pbar.set_postfix({
				'lr': f'{lr_schema}',
				'train_loss': f'{epoch_mean_train_loss_list[-1]:.16f}',
				'val_loss': f'{epoch_mean_eval_loss_list[-1]:.16f}',
			})

		torch.save(model.state_dict(), f'model_parameters/_[{self.dataset_name}]_.pth')

		# after training, saving the logger
		loss_dict = {'train_loss': epoch_mean_train_loss_list, 'eval_loss': epoch_mean_eval_loss_list}
		dataframe = pd.DataFrame(loss_dict)
		if Continue:
			dataframe.to_csv(f'experiment_data/_[{self.dataset_name}]_train_eval_loss_data.csv',sep=',',index=False,
							 encoding='utf-8', mode='a', header=False)
		else:
			dataframe.to_csv(f'experiment_data/_[{self.dataset_name}]_train_eval_loss_data.csv', sep=',', index=False,
							 encoding='utf-8')
		return output_list


	def model_output_distribution_figure(self, output:list[np.ndarray]):
		count = 0
		for data in tqdm(output, desc='picturing'):
			data = data.reshape(-1)
			fig, ax = plt.subplots(figsize=(10,10))
			ax.hist(data, bins=64)
			ax.set_title('model_output_distribution', fontsize=10, fontweight='bold', fontname='Times New Roman')
			ax.set_ylabel("Count", fontsize=12, fontweight='bold', fontname='Times New Roman')
			ax.tick_params(axis='x', labelrotation=0)
			plt.savefig(f'Numeric_Stimulation_experiment/tensor{count}.png')
			count += 1
			plt.close()

		output = np.asarray(output).reshape(-1)
		fig, ax = plt.subplots(figsize=(10, 10))
		ax.hist(output, bins=128)
		ax.set_title('model_output_distribution', fontsize=10, fontweight='bold', fontname='Times New Roman')
		ax.set_ylabel("Count", fontsize=12, fontweight='bold', fontname='Times New Roman')
		ax.tick_params(axis='x', labelrotation=0)
		plt.savefig(f'Numeric_Stimulation_experiment/AllOutputDistribution.png')
		plt.close()

	def model_train_eval_loss_figure(self, epochs:int):
		dataframe = pd.read_csv(f'experiment_data/_[{self.dataset_name}]_train_eval_loss_data.csv', sep=',', encoding='utf-8')
		train_loss_list = dataframe['train_loss'].to_list()[:epochs]
		theoretical_loss_list = self.exponential_decay(train_loss_list)
		iterations = np.arange(0, epochs)

		plt.figure(figsize=(10, 10), dpi=300)
		ax = plt.gca()

		# draw the theoretical curve and real curve
		ax.plot(iterations, theoretical_loss_list, label='Theoretical Curve', color='#8073ac', linewidth=4)
		if self.optimizer == 'gd':
			ax.plot(iterations, train_loss_list, color='#e08214', linewidth=4, label='Experimental Curve')
		else:
			ax.plot(iterations, train_loss_list, color='#e08214', linewidth=1, label='Experimental Curve')


		plt.title(f'{self.optimizer.upper()} Train Loss Curves', fontsize=32, fontname='Times New Roman')
		plt.xlabel('Iterations',fontsize=22)
		plt.ylabel('Loss',fontsize=22)
		plt.grid(True, linestyle='--', alpha=0.7)
		plt.legend(loc='upper right',fontsize=22)

		plt.savefig(f'experiment_result_figure/_[{self.dataset_name}]_{self.optimizer.upper()}_train_loss.pdf')
		plt.show()

	def exponential_decay(self, experimentLossValueNdarrayOrList):
		# calculate the theoretical curve
		theoreticalCurveCalculation = []
		for t in range(len(experimentLossValueNdarrayOrList)):
			if t == 0:
				theoreticalCurveCalculation.append(experimentLossValueNdarrayOrList[0])
			else:
				theoreticalCurveCalculation.append(experimentLossValueNdarrayOrList[0]*(0.9995556**t))
		return theoreticalCurveCalculation

	def MatrixBoundedVartification(self):
		stimulation_epochs = 1000
		max_element_list = []
		for size in [16]:
			for epoch in tqdm(range(stimulation_epochs), 'calculate the maximums of absolute gaussian matrix...'):
				gaussian_matrix = torch.randn(size=(size,size)).to('cuda')
				max_element = np.max(np.real(torch.linalg.eigvals(gaussian_matrix).cpu().detach().numpy()))
				max_element_list.append(max_element)
			# visualize maximum of absolute gaussian matrix distribution
			fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

			# draw box plot
			n, bins, patches = ax.hist(
				max_element_list,
				bins="fd",  # utilize Freedman-Diaconis rule
				alpha=0.7,  # transparent
				color='#6a51a3',
				edgecolor='#4a1486',
				linewidth=0.5
			)

			ax.legend(frameon=True, framealpha=0.9, edgecolor='gray', bbox_to_anchor=(1, 1))

			stats_text = f"examples: ${len(max_element_list)}$\n"
			stats_text += f"mean: ${np.mean(max_element_list):.3f}$\n"
			stats_text += f"std: ${np.std(max_element_list):.3f}$\n"
			stats_text += f"max: ${stats.tmax(np.array(max_element_list)):.3f}$\n"
			stats_text += f"min: ${stats.tmin(np.array(max_element_list)):.3f}$\n"
			stats_text += f"skewness: ${stats.skew(np.array(max_element_list)):.3f}$"

			props = dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.5)

			# set axis
			plt.xticks(fontsize=24)
			plt.yticks(fontsize=24)

			plt.tight_layout()
			plt.savefig(f'experiment_result_figure/MatrixBounded{size}.pdf')
			plt.show()

	def plot_model_weight_matrix_heatmap(self, model):
		pass

if __name__ == '__main__':
	m = MAIN(file_path='dataset/timeSeriesDataset/exchange_rate.csv',
			 columns=['a','b','c','d','e','f','g','OT'],
			 dataset_name='exchange_rate',
			 optimizer='gd')

	m.MatrixBoundedVartification()
	epochs = 3000
	output_list = m.train(sequence_len=256, n_head=4, lr_schema=1e-5, mini_batch_size=171, epochs=epochs, Continue=False)
	m.model_train_eval_loss_figure(epochs=epochs)

	# heatmap at experiment D
	colors = ['#542788', '#8073ac', '#b2abd2', '#d8daeb', '#fee0b6', '#fdb863', '#e08214', '#b35806']
	cmap1 = LinearSegmentedColormap.from_list("BlueRed", colors)
	tensors_order_dict = torch.load(f'model_parameters/_[ETTh1]_.pth')
	count = 0
	for c in tensors_order_dict.keys():
		count += 1
		if count >8:
			break
		plt.rcParams.update({'font.size': 24})
		matrix = tensors_order_dict[c].to('cpu').detach().numpy()
		plt.figure(figsize=(8, 7), dpi=300)
		im = plt.imshow(matrix, cmap=cmap1, vmin=np.min(matrix), vmax=np.max(matrix))
		cbar = plt.colorbar(im, orientation='vertical')
		plt.title(f'Weight{count} Heatmap', fontsize=24, fontname='Times New Roman')
		plt.savefig(f'experiment_result_figure/Weight{count}_Heatmap.pdf')
		plt.close()



