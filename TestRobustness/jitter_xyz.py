import numpy as np
import sys

from copy import deepcopy
	
def jitter_dat(xyzCoords, jitter_radius=10, jitter_barcode_perc=0.2):
	#randomVals = jitter_max * np.random.sample((xyzCoords.shape[1], xyzCoords.shape[2])) - jitter_max/2
	
	newXYZ = deepcopy(xyzCoords)

	m = xyzCoords.shape[0]
	bc = xyzCoords.shape[1]
	num_jitter = int(bc * jitter_barcode_perc)

	for i in range(m):
		jitterSample = jitter_radius * np.random.sample((bc,3)) - jitter_radius / 2
		idxs = list(range(bc))
		np.random.shuffle(idxs)
		jitterIdxs = idxs[:num_jitter]
		newXYZ[i,jitterIdxs,:] += jitterSample[jitterIdxs,:]	
	return newXYZ

def write_xyz(xyzs, tag):
	fp_xyz = open(tag + "_xyz.txt", "w")

	fp_xyz.write("x,y,z,Barcode,Cell\n")

	for i in range(xyzs.shape[0]):
		for b in range(xyzs.shape[1]):
			outstr = ','.join([str(x) for x in xyzs[i,b,:]]) + ',' + str(b) + ',' + str(i) + '\n'
			fp_xyz.write(outstr)

	fp_xyz.close()


def load_xyz(xyzfile):
	xyzfp = open(xyzfile, "r")
	header = list()
	datDict = dict()

	for line in xyzfp:
		if line.startswith('x'):
			header = line.rstrip().split(',')
			continue		
		else:
			x, y, z, barcode, cell = line.rstrip().split(',')	
			cell = int(cell)
			x = float(x)
			y = float(y)
			z = float(z)
			if cell not in datDict:			
				datDict[cell] = dict()
				datDict[cell]['dat'] = [[x,y,z]]
			else:
				datDict[cell]['dat'].append([x,y,z])

	xyzfp.close()	

	datAr = list()

	for c in datDict:
		currDat = datDict[c]['dat']
		currDat = np.array(currDat)
		datAr.append(currDat)

	datAr = np.array(datAr)

	return datAr

def main():
	if len(sys.argv) < 5:
		print("usage: " + sys.argv[0] + " <xyz file> <jitter radius> <% barcodes to jitter> <base tag>")
		sys.exit(1)

	xyzFile = sys.argv[1]
	jitterRad = float(sys.argv[2])
	jitterPerc = float(sys.argv[3])
	baseTag = sys.argv[4]
	xyzDat = load_xyz(xyzFile)

	jitterXYZ = jitter_dat(xyzDat, jitter_radius=jitterRad, jitter_barcode_perc=jitterPerc)
	write_xyz(jitterXYZ, baseTag + "_JitterRad-" + str(jitterRad) + "_jitterPerc-" + str(jitterPerc))

main()
