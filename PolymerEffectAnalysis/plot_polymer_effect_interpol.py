import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np

if len(sys.argv) != 3:
	print("usage: <PolymerEffectTrainSetTable.csv> <tag>")
	sys.exit(1)

filename = sys.argv[1]
tag = sys.argv[2]

fp = open(filename, "r")

header = True
lagAr = list()
exAr = list()
for line in fp:
	#print("On new line")
	if header == True:
		header = False
		continue

	lineAr = line.rstrip().split(',')
	lagValue = int(lineAr[0])
	examples = [ np.log10(float(x) / 100) for x in lineAr[1:] ]
	exAr.append(examples)
	#lagAr.append(lagValue)
	
fp.close()

exAr = np.array(exAr)
exArT = np.transpose(exAr)
exArL = exArT.tolist()

print("exArT shape: " + str(exArT.shape))
print("exArL length: " + str(len(exArL)))

fliercolor = dict(markerfacecolor='gray', markeredgecolor='black', marker='.', markersize=1)

fig = plt.figure()
plt.title("Autocorrelation values for different lags " + tag)
plt.boxplot(exArL, flierprops=fliercolor, medianprops=dict(color='k'))
plt.hlines(np.log10(1),0,26, color="gray", linestyles="dashed")
plt.ylabel("log10((Distance from Orig Polymer / Orig Polymer RG))")
plt.xlabel("Lag")

fig.savefig(tag + "_LogAutocorrPlot.pdf")

fig = plt.figure()
plt.title("Autocorrelation values for different lags " + tag)
plt.boxplot(exArL, flierprops=fliercolor, medianprops=dict(color='k'))
plt.hlines(100,0,26, color="gray", linestyles="dashed")
plt.ylim((0,500))
plt.ylabel("(Distance from Orig Polymer / Orig Polymer RG) * 100")
plt.xlabel("Lag")

fig.savefig(tag + "_AutocorrPlot_ylim_0_500.pdf")
