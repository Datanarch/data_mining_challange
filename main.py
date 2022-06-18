import os.path
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import Logs as lg
import time

import json
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sklearn.feature_selection as skF
import warnings
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind

def get_json_values(**args):
	value = ''
	files = json.load(open(os.path.join(root_path, args['jsonFile'])))[args['root']]
	NonFileName = [True if args.get('Name') is None else False]
	if args['root'] == "Files" and args.get('Name') is None:
		value = [os.getcwd() +'//'+ f['File Name']+'.'+f['Format'] if f['Current'] else f['Location'] +'//'+ f['File Name']+'.'+f[
		'Format'] for f in [file for file in files] if f['File Name'] == args['Value']][0]

	elif args['root'] == 'Feature Method Selection':
		value = [featureMethod[args['Value']] for featureMethod in files]

	elif args['root'] == "Top N Genes":
		value = [int(Gen[args['value']]) for Gen in files]

	elif args['root'] == 'Feature Properties':
		value = [int(limit[args['Value']]) for limit in files if limit['Name'] == args['Name']][0]

	elif args['root'] == 'Models':
		value = [Models[args['Value']] if Models[args['Value']] != 'KNN' else Models[args['Value']]+'-'+str(Models[
			'KNeighbors']) for Models in files]

	elif args['root'] == "Files" and args['Name'] == 'GetName':
		value = [f['File Name']+'.'+f['Format'] for f in [file for file in files] if f['File Name'] == args['Value']][0]

	else:
		raise TypeError("Value not supported")

	return value


# Top Genes classes
	topClassGenesPair = get_json_values(jsonFile='modelInteraction.Json', root='Top N Genes', value='Value')

	# # Some feature Properties to load first
	# FeatureProperties = get_json_values(jsonFile='modelInteraction.Json', root='Feature Properties')
	higherLimit = get_json_values(jsonFile='modelInteraction.Json', root='Feature Properties', Name='Higher Limit',
	                              Value='Value')
	lowerLimit = get_json_values(jsonFile='modelInteraction.Json', root='Feature Properties', Name='Lower Limit',
	                             Value='Value')

	foldLimit = get_json_values(jsonFile='modelInteraction.Json', root='Feature Properties',
	                            Name='Fold Difference Limit',
	                            Value='Value')

# loading project json configuration file
root_path = os.path.join(os.curdir, "Data")

# methodsSelection = [method['Name'] for method in mets]

# Top Genes classes
topClassGenesPair = get_json_values(jsonFile='modelInteraction.Json', root='Top N Genes', value='Value')

# # Some feature Properties to load first
# FeatureProperties = get_json_values(jsonFile='modelInteraction.Json', root='Feature Properties')
higherLimit = get_json_values(jsonFile='modelInteraction.Json', root='Feature Properties', Name='Higher Limit',
	                              Value='Value')

lowerLimit = get_json_values(jsonFile='modelInteraction.Json', root='Feature Properties', Name='Lower Limit',
	                             Value='Value')

foldLimit = get_json_values(jsonFile='modelInteraction.Json', root='Feature Properties', Name='Fold Difference '
	                                                                                              'Limit',
	                            Value='Value')


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
	classSet = read_data(fileName=get_json_values(jsonFile='modelInteraction.Json', root='Files', Name='GetName',
	                                              Value='Class'))
	trainSet = read_data(fileName=get_json_values(jsonFile='modelInteraction.Json', root='Files', Name='GetName',
	                                              Value='train'))
	testSet = read_data(fileName=get_json_values(jsonFile='modelInteraction.Json', root='Files', Name='GetName',
	 	                                              Value='test'))

	trainSet = data_threshold(data=trainSet, loweLimit=lowerLimit, HighLimit=higherLimit)
	trainSet = remove_fold(data=trainSet)
	trainSet = calculate_T_Value(data=trainSet, classSet=classSet, safeToFile=True)

	print(trainSet.head(10))


def read_data(path: str = root_path, fileName: str = "train.csv") -> pd.DataFrame:
	"""
	Given a path, folderName and fileName with extension type, return a pd.dataframe with the data
	:param path: default is current working directory
	:param fileName: default is Train.csv, but can read csv, txt
	:return: pd.Dataframe
	"""
	result = pd.DataFrame
	try:
		if fileName.split(".")[1] in ('csv', 'txt'):
			result = pd.read_csv(os.path.join(path, fileName))
			if fileName.split(".")[0] in ('train', 'Train', 'test', 'Test'):
				result.index = result['SNO']
				result.drop(labels='SNO', axis=1, inplace=True)
				result = result.transpose()
		else:
			raise TypeError("File format not allowed")

	except Exception as e:
		print(str(e))

	return result


def skew_classification(skew: float, type: str):
	"""classify according to the following skew values:
	Fairly Symmetrical  -0.5 to 0.5
	Moderate Skewed -0.5 to -1.0 and 0.5 to 1.0
	Highly Skewed< -1.0 and > 1.0
		1-Fairly
		2-Moderate
		3-Highly
	:param skew: skew value according to pd.Dataframe.Skew()
	:param type: string to identify what type to return {I-integer values; S-String values}
	:return: string classification
	"""
	returnVal = ''
	classification = { 'S':{1:'Fairly', 2:'Moderate', 3:'Fairly'},'I':{1:1, 2:2, 3:3}},
	try:
		if(float(skew) >= float(-0.5)) and (float(skew) <= float(0.5)):
			returnVal = classification[type][1]
		elif ((float(skew) >= float(-1.0))and (float(skew)) < float(-0.5)) or ((float(skew) > float(0.5))and (float(skew)) < float(1.0)):
			returnVal = classification[type][2]
		elif (float(skew) < float(-1.0)) or (float(skew) > float(1.0)):
			returnVal = classification[type][3]
	except Exception as ex:
		returnVal += 'Error:'+str(ex)

	return returnVal


def skew_sign(skew:float, type:str):
	"""

	:param skew:
	:param type:
	:return:
	"""
	returnVal = ''
	classification = {'S':{1:'+', 2:'-'}, 'I':{1:1, 2:-1}}
	try:
		if(float(skew) < 0):
			returnVal = classification[type][2]
		else:
			returnVal = classification[type][1]
	except Exception as ex:
		classification += 'Error:'+str(ex)
	return returnVal


def scatter_coefficient(X, normalize=True):
	corr = np.corrcoef(X, rowvar=False)
	if normalize:
		return np.linalg.det(corr) / np.trace(corr)
	else:
		return np.linalg.det(corr)


def psi_index(X, normalize=False):
	"""

	:param X:
	:param normalize:
	:return:
	"""
	corr = np.corrcoef(X, rowvar=False)
	# Eigenvalues and eigenvectors from the correlation matrix
	eig_val, eig_vec = np.linalg.eig(corr)
	idx = eig_val.argsort()[::-1]
	eig_val = eig_val[idx]
	if normalize:
		p = X.shape[0]
		return np.sum((eig_val - 1)**2) / (p*(p-1))
	else:
		return np.sum((eig_val - 1)**2)


def method_selection_function(**args):
	"""
	:param args:
	:return:
	"""
	if args['function'] == 'chi2':
		return chi2
	elif args['function'] == 'mutual_info_classif':
		return mutual_info_classif
	elif args['function'] == 'f_classif':
		return f_classif
	elif args['function'] == 'RandomForestClassifier':
		return RandomForestClassifier(n_estimators = 100)


def select_features(**args):
	"""

	:param args:
	:return:
	"""
	# configure to select all features
	fs = SelectKBest(score_func= method_selection_function(function= args['function']), k=args['features'])
	# learn relationship from training data
	fs.fit(args['X_train'], args['y_train'])
	# transform train input data
	X_train_fs = fs.transform(args['X_train'])
	# transform test input data\n",
	X_test_fs = fs.transform(args['X_test'])
	return X_train_fs, X_test_fs, fs


def data_threshold(data: pd.DataFrame, loweLimit: int = lowerLimit, HighLimit: int = higherLimit) -> pd.DataFrame:
	"""
	given the data, lower and higher limit, will delete the columns that go beyond given limits
	:param data: Dataframe data
	:param loweLimit: int with lower limit
	:param HighLimit: int wiht high limit
	:return: dataframe without lower and higher limits
	"""
	trainSet = data.copy(deep=True)
	data = data.to_numpy(copy=True)
	# Loop over genes with all samples to find the index of genes that do not have enough high or low limit
	genes_to_delete = [idx for idx, genes_row in enumerate(data.T) if np.max(genes_row[1:]) < loweLimit or np.min(
		genes_row[1:]) > HighLimit]
	# Deleting columns
	trainSet = trainSet[np.setxor1d(trainSet.columns, trainSet.columns[genes_to_delete])]
	return trainSet


def calculate_T_Value(data: pd.DataFrame, classSet: pd.DataFrame, safeToFile: bool = False) -> pd.DataFrame:
	"""
	Calculate the T-value and P-value per column
	:param data: Dataframe with data
	:param classSet: Dataframe with colum Class
	:param safeToFile: True/False to write into file
	:return: Dataframe with ['Class', 'Gene 1', 'Indices of Gene 1', 'Gene 2', 'Indices of Gene 2', 't-value',
	'p-value']
	"""
	encoder, classes, le = get_encoding(data=classSet, column='Class')
	df = pd.DataFrame()
	train = data.to_numpy()[:, :]
	# Placeholder for all class individual t test result
	total_t_result = []
	print(f'T-test Started on {len(set(classes))} class with {train.shape[1]} genes.\n')
	# Loop over all classes
	for cls in range(len(set(encoder))):
		print(f'T-test on Class: {le.inverse_transform((cls,))[0]}')
		# Append class-based results in other list
		cls_t_result = []
		# Get indices of classes
		samp = np.where(encoder == cls)[0] + 1
		# Take the first gene for t test
		for gene_0 in range(train.shape[1]):
			if np.any(train[1:, gene_0] < lowerLimit) or np.any(train[1:, gene_0] > higherLimit):
				continue
			# Calculate t and p values when testing with all the remaining genes
			for gene_1 in range(gene_0 + 1, train.shape[1]):
				if np.any(train[1:, gene_0] < lowerLimit) or np.any(train[1:, gene_0] > higherLimit):
					continue
				# Calculate t and p values when testing with all the remaining genes
				t_value, p_value = ttest_ind(train[samp, gene_0], train[samp, gene_1])
				cls_t_result.append((le.inverse_transform((cls,))[0], train[0, gene_0], gene_0, train[0, gene_1],
				                     gene_1, t_value, p_value))
				total_t_result.append(cls_t_result)

	# If desired, save these results in an additional file
		cols = ['Class', 'Gene 1', 'Indices of Gene 1', 'Gene 2', 'Indices of Gene 2', 't-value', 'p-value']
		data = np.squeeze(np.array(total_t_result).reshape((1, -1, 7)))
		df = pd.DataFrame(data, columns=cols)
		if safeToFile:
			path = os.path.join(root_path, "pp5i_t_result.gr.csv")
			df.to_csv(path, index=False)
	print('\nT-test completed!')

	return df


# def top_N_Values(Data: pd.DataFrame, topN: list = topClassGenesPair):
# 	for n in topN:
# 		n_train_list = []
# 		total_indices = []
# 		for encoded_class, cls_t_result in enumerate(t_test_result):


def plotting_values_result():
	# classet reading and formating class set dataframe
	classSet = read_data(fileName="class.txt")
	classSet.index = np.arange(len(classSet))+1

	# training set  dataframe: uploading and formatting
	trainSet = read_data(fileName="train.csv")
	trainSet.index = trainSet['SNO']
	trainSet.drop(labels='SNO', axis=1, inplace=True)
	trainSet = trainSet.transpose()

	testSet = read_data(fileName="test.csv")


	trainSet = data_threshold(data=trainSet, loweLimit=lowerLimit,HighLimit=higherLimit)
	trainSet = data_fold(data=trainSet, foldLimit=foldLimit)

	print("Showing Train data describe")
	print(trainSet.describe().transpose())

	# let first normalized the data so that way we do not have any feature that could make some noise and help the model to be faster and accurrate
	toScaleColumns = [col for col in trainSet.columns if col not in ('SNO','Class')]
	scaler = MinMaxScaler()
	df_normalized = pd.DataFrame(data=scaler.fit_transform(trainSet[toScaleColumns]), columns=toScaleColumns)
	df_normalized.dropna(inplace=True)
	print("Showing normalized data with Min-Max Scaler")
	print(df_normalized)

	print("All feature resume data")

	# checking nan values or no value in data
	columnsToWorkWith = [col for col in trainSet.columns.values if col not in('SNO', 'Class')]
	df_feature_selection =pd.DataFrame(index=columnsToWorkWith)
	df_feature_selection['HasNanValues'] = trainSet[columnsToWorkWith].apply(pd.isna, axis=0).any()
	df_feature_selection['Has0Values'] = (trainSet.values == 0).any(axis=0)
	df_feature_selection['HasNegValues'] = (trainSet.values < 0).any(axis=0)
	df_feature_selection['MinValue'] = trainSet.apply(np.min, axis=0)
	df_feature_selection['MaxValue'] = trainSet.apply(np.max, axis=0)
	df_feature_selection['Mean'] = trainSet.apply(np.mean, axis=0)
	df_feature_selection['Std'] = trainSet.apply(np.std, axis=0)
	df_feature_selection['Var'] = trainSet.apply(np.var, axis=0)
	df_feature_selection['1Quant(25%)'] = np.quantile(trainSet, 0.25, axis=0)
	df_feature_selection['3Quant(75%)'] = np.quantile(trainSet, 0.75, axis=0)
	print(df_feature_selection)

	print(" Display visually feature selection top N")
	X = df_normalized # trainSet[columnsToWorkWith]
	y = classSet['Class']
	result = pd.DataFrame(data=df_normalized.columns.values, columns=['Columns'])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

	print("Feature data dimension:", X.shape)
	for methodSelection in methodsSelection:
		for feature in topClassGenesPair:
			# feature selection process
			X_train_fs, X_test_fs, fs = select_features(function=methodSelection,
		                                            X_train=X_train,
		                                            y_train=y_train,
		                                            X_test=X_test,
		                                            features=feature
		                                            )
		# what are scores for the features
			result[f'{methodSelection}_top{feature}_score'] = fs.scores_
			result[f'{methodSelection}_top{feature}'] = fs.get_support()

		fig = plt.figure()
		ax = fig.add_axes([0,0,1,1])
		ax.set_ylabel('Score Value')
		ax.set_xlabel('Genes Selected')
		plt.xticks(rotation=90)
		plt.yticks(rotation=30)
		ax.set_title(f'Feature Selection top {feature} with {methodSelection}')
		ax.bar([i for i in result['Columns'][result[f'{methodSelection}_top{feature}'] == True]],
		       result[f'{methodSelection}_top{feature}_score'][result[f'{methodSelection}_top{feature}'] == True])
		plt.show()
		r = [i for i in result.columns if not '_score' in i and not 'Columns' in i]
		# classType  = [ i[0:str(i).find('_top')] for i in r]
		# topType = [ i[str(i).find('_top')+1:len(str(i))] for i in r]
	featureSelectionResult = pd.DataFrame()
	for classType, topType in zip([ i[0:str(i).find('_top')] for i in r], [ i[str(i).find('_top')+1:len(str(i))] for i in r]):
		featureSelectionResult= featureSelectionResult.append({'ClassificationType':classType,
		                                                       'TopType':topType,
		                                                       'ColumFeatures':result[result[f'{classType}_{topType}'] == True][[f'Columns']].values.tolist(),
		                                                       'FeaturesScore':result[result[f'{classType}_{topType}'] == True][[f'{classType}_{topType}_score']].values.tolist()
		                                                       },
		                                                      ignore_index=True)

	print("Displaying feature selection result according to score")
	print(featureSelectionResult)


def Model_Selection(**args):
	selector = None
	if args['Model'] == 'Gaussian':
		selector = GaussianNB()
	if args['Model'] == 'Decision tree':
		selector = DecisionTreeClassifier()
	if args['Model'] == 'KNN':
		selector = KNeighborsClassifier(n_neighbors=args['K'])
	if args['Model'] == 'MLPClassifier':
		selector = MLPClassifier(solver='lbfgs', random_state=1)
	return selector.fit(X=args['X'], y=args['y'])


def remove_fold(data: pd.DataFrame, fold_n: int = foldLimit) -> pd.DataFrame:
	"""
	Given a dataframe with data, return dataframe with columns that do not fit the fold number
	:param data: dataframe data
	:param fold_n: fold number to look for default: foldLimit
	:return: dataframe without fold columns
	"""
	trainSet = data.copy(deep=True)
	data = data.to_numpy(copy=True)
	# Loop over genes with all samples to find the index of genes that do not have enough fold
	genes_to_delete = [idx for idx, genes_row in enumerate(data.T) if np.max(genes_row[1:]) < fold_n * np.min(
		genes_row[1:])]
	# Delete gene columns from training and test data
	trainSet = trainSet[np.setxor1d(trainSet.columns, trainSet.columns[genes_to_delete])]

	return trainSet


def low_variance(data: pd.DataFrame, varianceValue: int):
	"""
	Given the data and min.Variance, will delete the column higher than the given value
	:param data: Dataframe to look up variance
	:param varianceValue: max variance
	:return:data without column with higher variance than given
	"""
	to_delete = [idx for idx, genes_row in enumerate(data.T) if np.std(genes_row[1:]) < varianceValue]

	data = data[np.setxor1d(data.columns, data.columns[to_delete])]
	return data


def get_encoding(data: pd.DataFrame, column: str = 'Class'):
	"""
	According to a dataframe, return encoder, class and labelEncoder
	:param data: dataframe with class
	:param column: column class name to use
	:return: encoder, class and labelEncoder object
	"""
	labelEncoder = LabelEncoder()
	classes = data[column].values
	encoder = labelEncoder.fit_transform(data[column])

	return encoder, classes, labelEncoder


if __name__ == "__main__":
	main()
