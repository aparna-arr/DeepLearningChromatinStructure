import sys
import numpy as np
import re
import glob 
import math

def main():
	if len(sys.argv) < 4:
		print("usage: process_polymers.py <tag> <num examples> <prob of random noise> <dir1 with state_block files> <dir2> <...>")
		sys.exit(1)

	tag = sys.argv[1]
	m = int(sys.argv[2])
	prob = float(sys.argv[3])
	dirlist = sys.argv[4:]

	perc_split = (0.6, 0.2, 0.2)
	m_train = math.floor(m * perc_split[0])
	m_dev = math.floor(m * perc_split[1])
	m_test = math.floor(m * perc_split[2])

	files = list()

	for d in dirlist:
		files.extend(glob.glob(d + '/*state_block*'))

	numFiles = len(files)
	numBarcodes = 52
	
	polymerXYZs = np.zeros([numFiles, numBarcodes, 3])
	distances = np.zeros([numFiles,1])
	
	currFile = 0

	for f in files:
		fp = open(f, 'r')

		currBarcode = 0
		searchPattern = re.compile('\s*<Position x.+')
		for line in fp:
			if searchPattern.search(line):
				valPattern = re.match(r'.*x="(.+)"\sy="(.*)"\sz="(.*)".*', line)
				x = float(valPattern.group(1))
				y = float(valPattern.group(2))
				z = float(valPattern.group(3))

				polymerXYZs[currFile,currBarcode,0] = x
				polymerXYZs[currFile,currBarcode,1] = y
				polymerXYZs[currFile,currBarcode,2] = z
							

				currBarcode += 1
				
		fp.close()

		dist = np.linalg.norm(polymerXYZs[currFile,20,:]-polymerXYZs[currFile,40,:])

		distances[currFile] = dist		

		currFile += 1

	np.random.seed(1)
	randomIdxs = np.arange(numFiles)
	np.random.shuffle(randomIdxs)

	distances = distances[randomIdxs,:]
	polymerXYZs = polymerXYZs[randomIdxs,:,:]

	distances = distances[:m,:]
	polymerXYZs = polymerXYZs[:m,:,:]

	sortedDistances = np.sort(distances,axis=0)	
	kthIdx = int(sortedDistances.shape[0] * 0.3)	
	kthVal = sortedDistances[kthIdx]

	print("kthVal: " + str(kthVal))

	rnas = np.zeros(distances.shape)
	for d in range(distances.shape[0]):
		if np.random.random_sample() < prob:
			rnas[d] = int(round(np.random.random_sample()))
		elif distances[d] < kthVal:
			rnas[d] = 1

	write_results(polymerXYZs[0:m_train,:,:], rnas[0:m_train,:], numBarcodes, "train_" + tag)
	write_results(polymerXYZs[m_train:m_train+m_dev,:,:], rnas[m_train:m_train+m_dev,:], numBarcodes, "dev_" + tag)
	write_results(polymerXYZs[m_train+m_dev:m_train+m_dev+m_test,:,:], rnas[m_train+m_dev:m_train+m_dev+m_test,:], numBarcodes, "test_" + tag)


def write_results(xyzs, rnas, numBarcodes, tag):
	fp_xyz = open(tag + "_xyz.txt", "w")
	fp_rna = open(tag + "_rna.txt", "w")

	fp_xyz.write("x,y,z,Barcode,Cell\n")
	fp_rna.write("RNA,Cell\n")

	for i in range(xyzs.shape[0]):
		for b in range(numBarcodes):
			outstr = ','.join([str(x) for x in xyzs[i,b,:]]) + ',' + str(b) + ',' + str(i) + '\n'
			fp_xyz.write(outstr)

		rnastr = str(int(rnas[i])) + ',' + str(i) + '\n'
		fp_rna.write(rnastr)

	fp_xyz.close()
	fp_rna.close()
	
main()
