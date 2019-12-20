import numpy as np
import sys

## code to shuffle the RNA values of the dev set examples, in order to verify whether the model
## was performing better than random

def permute_RNA(rnafile):
	'''Read in the RNA file, and randomly shuffle the RNA values among examples'''
	rnafp = open(rnafile, "r")
	
	rnaPermute = open(rnafile + ".SHUFFLE", "w")

	rnaheader = list()
	rnaDict = dict()
	rnaList = list()
	cellList = list()
	for line in rnafp:
		if not line[0].isdigit():
			rnaPermute.write(line)
			continue		
		else:
			rna1, cell = line.rstrip().split(',')
			rna1 = int(rna1)
			cell = int(cell)
	
			rnaList.append(rna1)
			cellList.append(cell)

	rnaList = np.array(rnaList)
	np.random.shuffle(rnaList)
	
	# write out the shuffled data		
	for i in range(rnaList.shape[0]):
		rnaPermute.write(str(rnaList[i]) + "," + str(cellList[i]) + "\n")

	rnafp.close()
	rnaPermute.close()

def main():
	if (len(sys.argv) < 2):
		print("usage: <label file to permute>\n", file=sys.stderr)
		sys.exit(1)

	permute_RNA(sys.argv[1])
main()	
