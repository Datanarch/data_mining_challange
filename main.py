import os.path
import pandas as pd
import Logs as lg

logFile = lg.LogFile(FileLocation=os.getcwd())


def clean_all_files(path: str, filesName: list = None):
	"""
	Clean all files before writing or creating new files from project execution
	:param filesName: list file to be deleted default None
	:param path: path to clean or delete
	:return: empty/print statement
	"""
	try:
		for file in os.listdir(path):
			if filesName is None:
				os.remove(os.path.join(path, file))
			else:
				if file in filesName:
					os.remove(os.path.join(path, file))

	except Exception as e:
		print(str(e))


def main():
	"""
	Main function
	:return: void
	"""
	trainSet = read_data(fileName="train.csv")
	testSet = read_data(fileName="test.csv")


def read_data(path: str = os.getcwd(), folderName: str = "Data", fileName: str = "train.csv") \
		-> pd.DataFrame:
	"""
	Given a path, folderName and fileName with extension type, return a pd.dataframe with the data
	:param path: default is current working directory
	:param folderName: default is Data directory
	:param fileName: default is Train.csv, but can read csv, txt
	:return: pd.Dataframe
	"""
	result = pd.DataFrame.empty
	if fileName.split(".")[1] in ('csv', 'txt'):
		result = pd.read_csv(os.path.join(path, folderName, fileName))
	else:
		raise TypeError("File format not allowed")
	logFile.write_to_file(f"File {fileName} error: File format not allowed")

	return result


if __name__ == "__main__":
	clean_all_files(path="C:\\Users\\henry\\Documents\\Bishops class\\3rd Semester\\CS-590 Masters Project\\Challege 1 - "
	                     "Data Mining - Dr. Layachi Bentabet\\DataMiningProject\\Prueba""",
	                filesName=["logFile.txt"])

	# logFile.write_to_file(Message="Starting Analysis")
	# main()
