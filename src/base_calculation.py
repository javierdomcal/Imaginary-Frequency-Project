from abc import ABC, abstractmethod

class BaseCalculation():
	def __init__(self, molecule)


	@abstractmethod
	def analytical_gradient_splitted(self):
		"""
		Perform the calculation specific for each calculation type
		"""
		pass

	def numerical_hessian_splitted(self):
		pass

		