import time
import datetime


class Timer():
	def __init__(self, num_steps, start_step=0):
		self.num_steps = num_steps
		self.start_step = start_step
		self.current_step = start_step
		self.start_time = time.time()
		self.elapsed_time = time.time()

	def step(self):
		self.current_step += 1

	def set_current_step(self, step):
		self.current_step = step

	def get_elapsed_time(self):
		self.elapsed_time = time.time() - self.start_time
		return str(datetime.timedelta(seconds=int(self.elapsed_time)))

	def get_estimated_time(self):
		self.elapsed_time = time.time() - self.start_time
		remaining_step = self.num_steps - self.current_step

		if self.current_step == self.start_step:
			return str(datatime.timedelta(seconds=int(0)))
		estimated_time = self.elapsed_time * remaining_step / (self.current_step - self.start_step)
		return str(datetime.timedelta(seconds=int(estimated_time)))

