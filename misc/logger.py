import os
import pandas
from datetime import datetime


class Logger():
	def __init__(self, path, num_epochs):
		assert num_epochs > 0
		self.path = path
		self.num_epochs = num_epochs
		self.time_stamp = self._get_time_stamp()

		if os.path.exists(path):
			self.df = pandas.read_csv(path)
			if self.df.shape[0] < num_epochs:
				self.df.reindex(index=range(1,num_epochs+1))
		else:
			self.df = pandas.DataFrame(index=range(1,num_epochs+1))

		self.df[self.time_stamp] = [None] * num_epochs

	def set(self, epoch, val):
		assert epoch >= 0 and epoch < self.num_epochs
		self.df[self.time_stamp][epoch] = val
		self.save()

	def save(self):
		self.df.to_csv(self.path)

	def _get_time_stamp(self):
		return datetime.now().strftime('%Y/%m/%d %H:%M:%S')
