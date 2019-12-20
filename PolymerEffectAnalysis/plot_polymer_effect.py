import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
	print("usage: <PolymerEffectTrainSetTable.csv>")
	sys.exit(1)

filename = sys.argv[1]

fp = open(filename, "r")

header = True
lagAr = list()
exAr = list()
for line in fp:
	if header == True:
		header = False
		continue

	lineAr = line.rstrip().split(',')
	lagValue = int(lineAr[0])
	examples = [ float(x) for x in lineAr[1:] ]
	exAr.append(examples)
	lagAr.append(lagValue)
	
fp.close()

fig = plt.figure()
plt.title("Autocorrelation values for different lags")
plt.boxplot(exAr)
plt.ylabel("Correlation Values")
plt.xlabel("Lag")

fig.savefig("AutocorrPlot.pdf")
