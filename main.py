import os.path

import pandas as pd


def main():
	"""
	Main function
	:return: void
	"""
	trainSet = read_data(fileName="train.csv")
	testSet = read_data(fileName="test.csv")


def read_data(path: str = os.getcwd(), folderName: str = "Data", fileName: str = "train.csv")\
		-> pd.DataFrame:
	"""
	Given a path, folderName and fileName with extension type, return a pd.dataframe with the data
	:param path: defualt is current working directory
	:param folderName: default is Data directory
	:param fileName: default is Train.csv, but can read csv, txt
	:return: pd.Dataframe
	"""
	result = pd.DataFrame.empty

	if fileName.split(".")[1] in('csv', 'txt'):
		result = pd.read_csv(os.path.join(path, folderName, fileName))
	else:
		raise TypeError("File format not allowed")

	return result


if __name__ == "__main__":
	main()