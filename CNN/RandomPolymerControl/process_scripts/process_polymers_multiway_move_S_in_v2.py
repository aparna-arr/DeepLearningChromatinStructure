import sys
import numpy as np
import re
import glob 
import math

def main():
	if len(sys.argv) < 4:
		print("usage: process_polymers_multiway.py <tag> <num examples> <dir1 with state_block files> <dir2> <...>")
		sys.exit(1)

	tag = sys.argv[1]
	m = int(sys.argv[2])
	dirlist = sys.argv[3:]

	perc_split = (0.6, 0.2, 0.2)
	m_train = math.floor(m * perc_split[0])
	m_dev = math.floor(m * perc_split[1])
	m_test = math.floor(m * perc_split[2])

	prob_silence = 0.8
	prob_e1_and_e2_on = 0.8
	prob_e1_or_e2_on = 0.1

	files = list()

	for d in dirlist:
		files.extend(glob.glob(d + '/*state_block*'))

	numFiles = len(files)
	numBarcodes = 52
	
	polymerXYZs = np.zeros([numFiles, numBarcodes, 3])
	e1p_distances = np.zeros([numFiles,1])
	e2p_distances = np.zeros([numFiles,1])
	sp_distances = np.zeros([numFiles,1])
	
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

		# Barcodes: 1-52
		# Promoter: 20
		# E1: 5
		# E2: 35
		# S: 50		

		# old distance for just one enhancer-promoter
		#dist = np.linalg.norm(polymerXYZs[currFile,20,:]-polymerXYZs[currFile,40,:])
		dist_e1p = np.linalg.norm(polymerXYZs[currFile,5,:]-polymerXYZs[currFile,20,:])
		dist_e2p = np.linalg.norm(polymerXYZs[currFile,35,:]-polymerXYZs[currFile,20,:])
		dist_sp = np.linalg.norm(polymerXYZs[currFile, 45,:]-polymerXYZs[currFile,20,:])

		e1p_distances[currFile] = dist_e1p		
		e2p_distances[currFile] = dist_e2p		
		sp_distances[currFile] = dist_sp		

		currFile += 1

	np.random.seed(1)
	randomIdxs = np.arange(numFiles)
	np.random.shuffle(randomIdxs)

	e1p_distances = e1p_distances[randomIdxs,:]
	e2p_distances = e2p_distances[randomIdxs,:]
	sp_distances = sp_distances[randomIdxs,:]
	polymerXYZs = polymerXYZs[randomIdxs,:,:]

	e1p_distances = e1p_distances[:m,:]
	e2p_distances = e2p_distances[:m,:]
	sp_distances = sp_distances[:m,:]
	polymerXYZs = polymerXYZs[:m,:,:]

	# E1 - P
	e1p_sortedDistances = np.sort(e1p_distances,axis=0)	
	e1p_kthIdx = int(e1p_sortedDistances.shape[0] * 0.60)	
	e1p_kthVal = e1p_sortedDistances[e1p_kthIdx]

	e1p_contact = (e1p_distances < e1p_kthVal)

	# E2 - P
	e2p_sortedDistances = np.sort(e2p_distances,axis=0)	
	e2p_kthIdx = int(e2p_sortedDistances.shape[0] * 0.60)	
	e2p_kthVal = e2p_sortedDistances[e2p_kthIdx]

	e2p_contact = (e2p_distances < e2p_kthVal)

	# S - P
	sp_sortedDistances = np.sort(sp_distances,axis=0)	
	sp_kthIdx = int(sp_sortedDistances.shape[0] * 0.30)	
	sp_kthVal = sp_sortedDistances[sp_kthIdx]

	sp_contact = (sp_distances < sp_kthVal)

	#prob_silence = 0.8
	#prob_e1_and_e2_on = 0.8
	#prob_e1_or_e2_on = 0.1

	rnas = np.zeros([m,1])

	for d in range(m):
		if np.random.random_sample() < prob_silence:
			# follow silencing rule
			# silence if S-P
			if sp_contact[d]:
				rnas[d] = 0
			else:
				# not specifically silenced
				
				if np.random.random_sample() < prob_e1_and_e2_on:
					# follow e1 & e2 synergy rule
					# on if E1-P-E2
					if e1p_contact[d] and e2p_contact[d]:
						rnas[d] = 1
					else:
						if np.random.random_sample() < prob_e1_or_e2_on:
							# follow e1 OR e2 rule
							# on if E1-P or E2-P
							if e1p_contact[d] or e2p_contact[d]:
								rnas[d] = 1
							else:
								# cell is off
								rnas[d] = 0
						else:
							# not following e1 or e2 rule
							# get a random result
							# or off??
							#rnas[d] = int(round(np.random.random_sample()))
							# still following synergy rule!
							rnas[d] = 0
				else:
					# not following synergy rule
					# could still follow e1 or e2 rule
					if np.random.random_sample() < prob_e1_or_e2_on:
						# follow e1 OR e2 rule
						# on if E1-P or E2-P
						if e1p_contact[d] or e2p_contact[d]:
							rnas[d] = 1
						else:
							# cell is off
							rnas[d] = 0
					else:
						# not following e1 or e2 rule
						# still following silencer rule
						# get a random result
						# because "no silencer" does not mean "on"
						rnas[d] = int(round(np.random.random_sample()))
						
						
		else:
			# not following silencing rule
			# could still follow synergy rule OR E1 & E2 rule?
			#rnas[d] = int(round(np.random.random_sample()))
			if np.random.random_sample() < prob_e1_and_e2_on:
				# follow e1 & e2 synergy rule
				# on if E1-P-E2
				if e1p_contact[d] and e2p_contact[d]:
					rnas[d] = 1
				else:
					if np.random.random_sample() < prob_e1_or_e2_on:
						# follow e1 OR e2 rule
						# on if E1-P or E2-P
						if e1p_contact[d] or e2p_contact[d]:
							rnas[d] = 1
						else:
							# cell is off
							rnas[d] = 0
					else:
						# not following e1 or e2 rule
						# still following synergy rule
						# get a random result -- NO
						#rnas[d] = int(round(np.random.random_sample()))
						# OFF
						rnas[d] = 0
			else:
				# not following synergy rule
				# could still follow e1 or e2 rule
				if np.random.random_sample() < prob_e1_or_e2_on:
					# follow e1 OR e2 rule
					# on if E1-P or E2-P
					if e1p_contact[d] or e2p_contact[d]:
						rnas[d] = 1
					else:
						# cell is off
						rnas[d] = 0
				else:
					# not following e1 or e2 rule
					# get a random result
					rnas[d] = int(round(np.random.random_sample()))


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
