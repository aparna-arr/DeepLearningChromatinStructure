import sys
from .ReadData import *

class ModelDriver():
	def __init__(self, params):
		self.params = params
		self.data = None
		self.model = None

	def polymer_effect(self):
		dat = ReadData(self.params['X_train'], self.params['Y_train'], self.params['X_dev'], self.params['Y_dev'], False, True, normalize=False).load_data()
	
		trainDat = dat['X_train'].reshape(dat['X_train'].shape[0], dat['X_train'].shape[1], dat['X_train'].shape[2])
		
		mTrain = trainDat.shape[0]

		endLag = 35
		
		lagValues = list()
		autoCorrValues = list()

		for lag in range(1,endLag):
			autoCorrValues.append(list())
			lagValues.append(lag)
			for ex in range(mTrain):
				currMap = trainDat[ex,:,:]
				autocorrMean = 0
				autocorrCount = 0
				for i in range(currMap.shape[0]):

					if i+lag >= currMap.shape[0]:
						break

					firstRow = currMap[i,:]
					secondRow = currMap[i+lag,:]
			
					# autocorrelate (partial ... )
					autocorr = np.corrcoef(firstRow, secondRow)[0][1]
					autocorrMean += autocorr
					autocorrCount += 1
					# add to list
				currListIdx = len(lagValues)-1
				autoCorrValues[currListIdx].append(autocorrMean/autocorrCount)	

		# print out as table
		outfileName = "PolymerEffectTrainSetTable.csv"
		fp = open(outfileName,"w")
		fp.write("Lag,Mean,Examples\n")

		for lagIdx in range(len(lagValues)):
			avgVal = np.mean(autoCorrValues[lagIdx])
			values = ",".join([str(x) for x in autoCorrValues[lagIdx]])
			lagStr = str(lagValues[lagIdx])
						
			fp.write(lagStr + "," + str(avgVal) + "," + values + "\n")

		fp.close()
				
