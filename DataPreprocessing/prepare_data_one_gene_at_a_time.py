import numpy as np
import sys
import math
from scipy import interpolate
from scipy.spatial.transform import Rotation as R

## This code preprocesses the raw experimental data into input for the model
## Also contains functions for normalization, centering, and data augmentation
## not all of which are useful now that I have moved to contact matricies for the most part

def read_data(xyzfile, rnafile):
	'''Function to read in raw experimental data, clean up NANs, and output useful data matricies'''
	xyzfp = open(xyzfile, "r")
	header = list()
	datDict = dict()

	# loop over the file
	for line in xyzfp:
		if not line[0].isdigit():
			header = line.rstrip().split(',')
			continue		
		else:
			# grab the useful information
			# "cell" being example number
			barcode, x, y, z, cell = line.rstrip().split(',')	
		
			barcode = int(barcode)
			cell = int(cell)

			# convert NaN strings to numpy nans
			if x == 'NaN':
				x = np.nan
			else:
				x = float(x)
			
			if y == 'NaN':
				y = np.nan
			else:
				y = float(y)
		
			if z == 'NaN':
				z = np.nan
			else:
				z = float(z)

			# add xyz coordinates by barcodes to data dictionary by cell/example number
			if cell not in datDict:			
				datDict[cell] = dict()
				datDict[cell]['dat'] = [[x,y,z]]
			else:
				datDict[cell]['dat'].append([x,y,z])

	xyzfp.close()	

	# read in the RNA data, or Y matrix
	rnafp = open(rnafile, "r")

	rnaheader = list()
	rnaDict = dict()

	for line in rnafp:
		if not line[0].isdigit():
			rnaheader = line.rstrip().split(',')
			continue		
		else:
			# get the 3 RNA values per example
			rna1, rna2, rna3, cell = line.rstrip().split(',')
			rna1 = int(rna1)
			rna2 = int(rna2)
			rna3 = int(rna3)
			cell = int(cell)

			rnaDict[cell] = dict()
			
			# add to the data dictionary
			rnaDict[cell]['dat'] = [rna1, rna2, rna3]

	rnafp.close()

	return datDict, rnaDict

def interpolate_coords(coords):
	'''function to interpolate NaN values as a minimal imputation strategy'''
	newcoords = coords
	nanidx = np.argwhere(np.isnan(coords))
	goodidx = np.argwhere(~np.isnan(coords))

	# identify "good" and "nan" values and indexes
	x = np.arange(len(coords))
	goodx = x[goodidx]
	nanx = x[nanidx]
	goody = coords[goodidx]
	goodx = goodx.squeeze()
	goody = goody.squeeze()

	# perform the interpolation
	f = interpolate.interp1d(goodx, goody, fill_value="extrapolate")

	ynew = f(nanx)
	
	# replace nan values with interpolated coordinates
	newcoords[nanx] = ynew

	return newcoords

def normalize(xyzdat):
	'''function to center XYZ coordinates on the center of mass'''
	normdat = xyzdat - np.mean(xyzdat, axis=0)
	return normdat

def rotate_data(xyzCoords, seed=1):
	'''function for data augmentation: randomly rotate the data slightly'''
	np.random.seed(seed)
	rotation = R.from_euler('xyz', [np.random.random(20), np.random.random(20), np.random.random(20)], degrees=True)
	
	return rotation.apply(xyzCoords)

def filter_dat(xyzdat, rnadat, req_xyz_perc = 0.5, interpol=True, norm=True, bool_thresh=-1):
	'''given the high rate of NaNs in the data, need to filter out cells with less real data than a useful theshold'''
	filtXYZDat = list()
	filtRNADat = list()
	
	# iterate over cells
	for c in xyzdat:
		currDat = xyzdat[c]['dat']
		currDat = np.array(currDat)
		sums = np.sum(currDat, axis=1)
		goodRows = np.sum(~np.isnan(sums))

		# keep only those cells/examples where we have more than the required real data percentage
		if goodRows / currDat.shape[0] >= req_xyz_perc:
			if interpol == True:
				# interpolate the missing values
				currDat[:,0] = interpolate_coords(currDat[:,0])
				currDat[:,1] = interpolate_coords(currDat[:,1])
				currDat[:,2] = interpolate_coords(currDat[:,2])

				if norm == True:
					# center the coordinates to the center of mass
					filtXYZDat.append(normalize(currDat))
				else:
					filtXYZDat.append(currDat)		
			else:
				if norm == True:
					filtXYZDat.append(normalize(currDat))
				else:
					filtXYZDat.append(currDat)		

			# make the output value binary
			currRNA = rnadat[c]['dat']
			currRNA = np.array(currRNA)

			if bool_thresh == -1:
				filtRNADat.append(currRNA)
			else:
				boolRNA = np.where(currRNA >= bool_thresh, 1, 0)
				filtRNADat.append(boolRNA)
	
	filtXYZDat = np.array(filtXYZDat)
	filtRNADat = np.array(filtRNADat)
	
	return filtXYZDat, filtRNADat

def write_file_one_RNA(xyz, rna, tag):
	'''write out the xyz and RNA data to a flattened format suitable for a file'''
	xyzfp = open(tag + "_xyz.txt", "w")
	rna1fp = open(tag + "_rna.txt", "w")

	xyzfp.write("x,y,z,Barcode,Cell\n")
	rna1fp.write("RNA,Cell\n")

	for c in range(xyz.shape[0]):
		currDat = xyz[c,:,:]
		currRNA = rna[c,:].tolist()
	
		strRNA1 = str(int(currRNA[0])) + ',' + str(c) + "\n"
		rna1fp.write(strRNA1)	

		for b in range(currDat.shape[0]):
			currXYZ = currDat[b,:].tolist()
			writeDat = currXYZ
			writeDat.append(b)
			writeDat.append(c)
			strXYZ = ",".join([str(x) for x in writeDat]) + "\n"
			xyzfp.write(strXYZ)
					
	xyzfp.close()
	rna1fp.close()


def balance_classes_and_write(xyzCoords, rna, seed, tag):
	'''balance classes either by augmenting the data with random rotations, or reducing the larger class to the smaller class size'''
	# iterate over the 3 genes
	for r in range(3):
		off = list()
		on = list()

		# figure out which examples have 0s or 1s for this gene
		for m in range(rna.shape[0]):
			if rna[m,r] == 0:
				off.append(m)
			else:

				on.append(m)

		# for now, not using data augmentation. 
		# Reduce the size of the larger OFF class to the size of the smaller ON class
		minCount = len(on)

		np.random.seed(seed)
		currXYZ = np.zeros((minCount * 2, xyzCoords.shape[1], xyzCoords.shape[2]))
		currRNA = np.zeros((minCount * 2, 1))

		offIdx = np.array(off)		
		np.random.shuffle(offIdx)
		
		newIdx = offIdx[0:minCount]

		currXYZ[0:minCount,:,:] = xyzCoords[np.array(on),:,:]
		currRNA[0:minCount,:] = rna[np.array(on),r].reshape((minCount,1))
		currXYZ[minCount:minCount*2,:,:] = xyzCoords[newIdx,:,:]
		currRNA[minCount:minCount*2,:] = rna[newIdx,r].reshape((minCount,1))

		totalIdx = np.arange(minCount*2)
		np.random.shuffle(totalIdx)

		write_file_one_RNA(currXYZ[totalIdx,:,:], currRNA[totalIdx,:], "train_one_gene_balance_classes_RNA_" + str(r) + "_" + tag)

def make_train_dev_test(xyzfilt, rnafilt, seed, tag, perc_split = (0.8, 0.1, 0.1), norm_dataset=False, balance=True):
	'''split the dataset into train/dev/test, and balance the dataset if required'''
	m = xyzfilt.shape[0]
	m_train = math.floor(m * perc_split[0])
	m_dev = math.floor(m * perc_split[1])
	m_test = math.floor(m * perc_split[2])

	# randomly shuffle and split into train/dev/test
	idxs = np.arange(m)
	np.random.seed(seed)
	np.random.shuffle(idxs)

	train_idx = idxs[0:m_train]
	dev_idx = idxs[m_train:m_train+m_dev]
	test_idx = idxs[m_train+m_dev:m_train+m_dev+m_test]

	trainDat = xyzfilt[train_idx,:,:]
	devDat = xyzfilt[dev_idx,:,:]
	testDat = xyzfilt[test_idx,:,:]

	trainRNA = rnafilt[train_idx,:]
	devRNA = rnafilt[dev_idx,:]
	testRNA = rnafilt[test_idx,:]

	if balance:
		balance_classes_and_write(trainDat, trainRNA, seed, tag)
		write_file(devDat, devRNA, "dev_one_gene_balance_classes_" + tag)
		write_file(testDat, testRNA, "test_one_gene_balance_classes_" + tag)
	else:
		write_file(trainDat, trainRNA, "train_" + tag)
		write_file(devDat, devRNA, "dev_" + tag)
		write_file(testDat, testRNA, "test_" + tag)

def main():
	if len(sys.argv) < 4:
		print("usage: <XYZ csv file> <RNA values csv file> <shuffle seed> <tag>", file=sys.stderr)
		sys.exit(1)
	
	xyzfile = sys.argv[1]
	rnafile = sys.argv[2]
	seed = int(sys.argv[3])
	tag = sys.argv[4]
	

	xyzdat, rnadat = read_data(xyzfile, rnafile)
	xyzfilt, rnafilt = filter_dat(xyzdat, rnadat, req_xyz_perc = 0.5, interpol=True, norm = True, bool_thresh = 1)
	make_train_dev_test(xyzfilt, rnafilt, seed, tag, perc_split=(0.6,0.2,0.2))	

main()
