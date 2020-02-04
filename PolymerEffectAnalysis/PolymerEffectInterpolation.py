import numpy as np
import sys
from scipy import interpolate

def read_data(xyzfile):
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

def get_norm_dat(rawDat):
	normDat = rawDat - np.mean(rawDat, axis=0)[None,:]
	return normDat

def calculate_RG(dat):
	rg = np.sqrt(np.sum(np.var(np.array(get_norm_dat(dat)),0)))
	return rg

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

def calculate(xyzDat, maxLag=25):
	lags = range(0,maxLag)
	
	lagResultsFull = list()
	lagResultsMean = list()

	RGs = list()
	for c in range(xyzDat.shape[0]):
		curr_rg = calculate_RG(xyzDat[c,:,:])
		#print("curr RG for cell " + str(c) + " RG " + str(curr_rg))
		RGs.append(curr_rg)

	for lag in lags:
		lagResultsFull.append(list())
		lagResultsMean.append(list())
		print("lag is " + str(lag+1))

		for c in range(xyzDat.shape[0]):
			lagResultsFull[lag].append(list())

			#print("c is " + str(c))
			#for na_start in range(1,52-(lag+1),(lag+1)):
			for na_start in range(0,52-(lag),(lag+1)):
			#for na_start in range(0,52-(lag),1):
				#print("na_start is " + str(na_start))
				copyPolymer = np.copy(xyzDat[c,:,:])
				
				napos_li = list()
				for na_pos in range(na_start,na_start+(lag+1)):
					copyPolymer[na_pos,0] = np.nan	
					copyPolymer[na_pos,1] = np.nan	
					copyPolymer[na_pos,2] = np.nan
				
					napos_li.append(na_pos)
		
				#napos_li = np.array(napos_li)	

				copyPolymer[:,0] = interpolate_coords(copyPolymer[:,0])	
				copyPolymer[:,1] = interpolate_coords(copyPolymer[:,1])	
				copyPolymer[:,2] = interpolate_coords(copyPolymer[:,2])
				
				interpolPolymer = np.copy(copyPolymer[napos_li,:])
				origPolymer = np.copy(xyzDat[c,napos_li,:])

				#print("origPolymer:")
				#print(origPolymer.shape)
				#print("interpolPolymer")
				#print(interpolPolymer.shape)

				squared_dist = np.sum((origPolymer - interpolPolymer)**2, axis=1)
				#print("size squared_dist " + str(squared_dist.shape))
				dists = np.sqrt(squared_dist)
				#print("dists len is " + str(dists.shape))
				normed = np.multiply(np.divide(dists, RGs[c]),100)
				#print("RG is " + str(RGs[c]))
				#print("dists")
				#print(dists)
				#print("normed")
				#print(normed)		
				#lagResults[lag].append([x for x in dists])
				#lagResultsFull[lag][c].extend([x for x in dists])
				lagResultsFull[lag][c].extend([x for x in normed])
				#lagResults[lag].append(np.mean(dists))
	
				#print("Lag is " + str(lag+1) + " c is " + str(c) + " len lagResults[lag] is " + str(len(lagResults[lag])) + " na_start is " + str(na_start))	
		
			lagResultsMean[lag].append(np.mean(lagResultsFull[lag][c]))
			#print("Lag is " + str(lag+1) + " cell " + str(c) + " len lagResultsFull[lag][c] is " + str(len(lagResultsFull[lag][c])) + " lagResultsMean[lag] len " + str(len(lagResultsMean[lag])))	
			#print("Lag is " + str(lag+1) + " cell " + str(c) + " len lagResultsFull[lag][c] is " + str(len(lagResultsFull[lag][c])) + " lagResultsMean[lag][c] value " + str(lagResultsMean[lag][c]))	
				
	lagResultsMean = np.array(lagResultsMean)
	print("size lagResults: " + str(lagResultsMean.shape))
	return lagResultsMean
					
def print_results(res, tag):
	outfilename = tag + "_results.txt"
	
	fp = open(outfilename, "w")
	header = "CellNum," + ",".join(["Lag_" + str(x) for x in range(res.shape[0])]) + "\n"
 
	fp.write(header)
	for c in range(res.shape[1]):
		lagResultsThisCell = list()

		for lag in range(res.shape[0]):
			lagResultsThisCell.append(res[lag][c] + 1)

		fp.write(str(c) + "," + ",".join([str(x) for x in lagResultsThisCell]) + "\n")
		#print("len of lag:" + str(len(res[lag])))
		#fp.write(str((lag+1)) + ',' + ",".join([str(x) for x in res[lag]]) + "\n")

	fp.close()

def main():
	if len(sys.argv) < 3:
		print("usage: <clean xyz text file> <tag>", file=sys.stderr)
		sys.exit(1)

	xyzfile = sys.argv[1]
	tag = sys.argv[2]

	xyzDat = read_data(xyzfile)
	results = calculate(xyzDat)

	print_results(results, tag)

main()
