import csv #csv module
import random
import math
def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))#read dataset
	dataset = list(lines)  #list datatype
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))#generate a random number
		trainSet.append(copy.pop(index))#pop this number in copy set and append it into trainSet
	return [trainSet, copy]
	
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[0] not in separated):
			separated[vector[0]] = []
		separated[vector[0]].append(vector)
	return separated
	
def mean(numbers):
	return sum(numbers)/float(len(numbers))
	
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[0]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
 
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

 	
def main():
 # first test case for the loadCsv function  result:pass
    filename = 'pima-indians-diabetes.csv'
    dataset = loadCsv(filename)
    print(dataset[1])
 #test case for spiltDataset function  result:pass
    #dataset = [[1], [2], [3], [4], [5]]
    #splitRatio = 0.67
    #train, test = splitDataset(dataset, splitRatio)
    #print(train)
    #print(test)
#test case for separateByClass function  result:pass
    #dataset = [[1,20,2], [2,21,0], [3,22,1]]
    #separated = separateByClass(dataset)
    #print(separated)
#test case for mean and stdev function  result:pass	
    #numbers = [1,2,3]
    #print(mean(numbers))
    #print(stdev(numbers))
#test case for summarize function  result:pass		
    #dataset = [[1,20,0], [2,21,1], [3,22,0], [1,20,0]]
    #summary = summarize(dataset)
    #print(summary)
#test case for summarizeByClass function  result:pass
    #dataset = [[1,20,1], [2,21,1], [1,22,1], [2,22,0],[3,10,9],[3,11,10]]
    #summary = summarizeByClass(dataset)
    #print(summary)
#test case for calculateProbability function  result:pass
    x = 0
    mean = 0
    stdev = 0.000001
    probability = calculateProbability(x, mean, stdev)
    print(probability)
#test case for calculateClassProbabilities function  result:pass
    #summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
    #inputVector = [1.1, 0.5]
    #probabilities = calculateClassProbabilities(summaries, inputVector)
    #print(probabilities)
#test case for predict function  result:pass	
    #summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
    #inputVector = [1.1, '?']
    #result = predict(summaries, inputVector)
    #print(result)
#test case for getpredict function  result:pass
    #summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
    #testSet = [[1.1, '?'], [19.1, '?']]
    #predictions = getPredictions(summaries, testSet)
    #print(predictions)
#test case for getAccuracy function  result:pass
    #testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
    #predictions = ['a', 'a', 'a']
    #accuracy = getAccuracy(testSet, predictions)
    #print(accuracy)
    #filename = 'pima-indians-diabetes.csv'
    #splitRatio = 0.67
    #dataset = loadCsv(filename)
    #trainingSet, testSet = splitDataset(dataset, splitRatio)
    #print(len(dataset), len(trainingSet), len(testSet))
    # prepare model
    #summaries = summarizeByClass(trainingSet)
	# test model
    #predictions = getPredictions(summaries, testSet)
    #accuracy = getAccuracy(testSet, predictions)
    #print(accuracy)

main()
