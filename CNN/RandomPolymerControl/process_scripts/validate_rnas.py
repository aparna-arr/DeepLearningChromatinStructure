import sys
import numpy as np

def main():
	if len(sys.argv) < 2:
		print("usage: validate_rnas.py <rna.txt file>")
		sys.exit(1)

	filename = sys.argv[1]
		
	fp = open(filename,"r")

	header = True
	total_rnas = list()

	for line in fp:
		if header:
			header = False
			continue
		rna, cell = line.rstrip().split(",")
		rna = int(rna)
		total_rnas.append(rna)

	fp.close()

	total_rnas = np.array(total_rnas)
	total_on = np.sum((total_rnas > 0))	
	total_off = np.sum((total_rnas < 1))

	print("For file " + filename + " counts are:")
	print("ON : " + str(total_on))
	print("OFF: " + str(total_off))
	print("Fraction on: " + str((total_on / (total_on + total_off))))	

main()
