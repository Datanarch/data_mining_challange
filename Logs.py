from datetime import datetime as dt
import os


class LogFile:
	"""
	Write a log file from coding interaction. It will help to understand how the model works
	and helps to see if there are any error while running
	"""

	def __init__(self, FileLocation: str = os.getcwd(), FileName: str = "logFile"):
		"""
		File log initilization, creates and write down the first line on file ( Log file creation)
		and initilizaed line counter
		:param str FileLocation: File path to be write default: code current object
		:param str FileName: File name dafault:logFile with timestamp format yyyy-mm-dd_hh:mm:ss
		:returns FileLog Object
		"""
		self.FilePath = FileLocation + '/Data/'
		# {dt.strftime(dt.today(), '%Y%m%d_%H%M%S')}
		self.FileName = str(f"{FileName}.txt")
		self.FileLine = 1
		self.write_to_file("Log File Creation")

	def write_to_file(self, Message):
		"""
		Write down a messange into file log already created
		:param str Message: Message to be written on file
		:return: None
		"""
		try:
			with open(os.path.join(self.FilePath, self.FileName), "a+") as file:
				file.write(f"|Line|:{self.FileLine}.....|Message|:{Message}.....|on|:{dt.now()}\n")
				self.counter_increment()
		except Exception as e:
			print(str(e))

	def write_report(self, Message):
		try:
			with open(os.path.join(self.FilePath, self.FileName), "a+") as file:
				file.write(f"|Line|:{self.FileLine}***********Report Start***********:{dt.now()}\n")
				file.write(f"{Message}\n")
				self.counter_increment()
				file.write(f"|Line|:{self.FileLine}***********Report End***********:{dt.now()}\n")
				self.counter_increment()
		except Exception as e:
			print(str(e))

	# def counter_increment(self):
	# 	"""
	# 	Increment line number on file log interaction
	# 	:return: int
	# 	"""
	# 	self.FileLine = self.FileLine + 1

	def counter_increment(self, increment=1):
		self.FileLine = self.FileLine + increment
