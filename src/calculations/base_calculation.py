from abc import ABC, abstractmethod

class BaseCalculation(ABC):
	def __init__(self, molecule)


	@abstractmethod
	def analytical_gradient_splitted(self):
		"""
		Perform the calculation specific for each calculation type
		"""
		pass

	@abstractmethod
	def numerical_hessian_splitted(self):
		pass

