import os.path
import pandas as pd
import Logs as lg

# Default data directory to read and write data
workingDirectory = os.path.join(os.getcwd(), "Data")


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
	print(trainSet)
	print(testSet)


def read_data(path: str = workingDirectory, fileName: str = "train.csv") -> pd.DataFrame:
	"""
	Given a path, folderName and fileName with extension type, return a pd.dataframe with the data
	:param path: default is current working directory
	:param folderName: default is Data directory
	:param fileName: default is Train.csv, but can read csv, txt
	:return: pd.Dataframe
	"""
	result = pd.DataFrame.empty
	if fileName.split(".")[1] in ('csv', 'txt'):
		result = pd.read_csv(os.path.join(path, fileName))
		logFile.write_to_file(f"File {fileName} reading Done")
	else:
		logFile.write_to_file(f"File {fileName} error: File format not allowed")
		raise TypeError("File format not allowed")

	return result


# cleaning all the information in case it already exists
clean_all_files(path=workingDirectory, filesName=["logFile.txt"])

logFile = lg.LogFile(FileLocation=workingDirectory)


if __name__ == "__main__":
	logFile.write_to_file(f"Starting Analysis")
	main()
