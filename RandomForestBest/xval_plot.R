library(ggplot2)
library(reshape2)

## Ubx ##
bestRFUbx<-"num-estimators_900_min-sample-split_15_max-depth_None_max-leaf-nodes_3_random-state_0_class-weight_balanced_rf-Ubx_KfoldXval.log"

## AbdA ##
bestRFAbdA<-"num-estimators_900_min-sample-split_15_max-depth_None_max-leaf-nodes_3_random-state_0_class-weight_balanced_rf-AbdA_KfoldXval.log"

## AbdB ##
bestRFAbdB<-"num-estimators_900_min-sample-split_15_max-depth_None_max-leaf-nodes_3_random-state_0_class-weight_balanced_rf-AbdB_KfoldXval.log"

########## CODE ############

bestRFUbxDat<-as.numeric(read.delim(bestRFUbx, header=F, sep=",")[1,])
bestRFAbdADat<-as.numeric(read.delim(bestRFAbdA, header=F, sep=",")[1,])
bestRFAbdBDat<-as.numeric(read.delim(bestRFAbdB, header=F, sep=",")[1,])


df<-data.frame(AbdA = bestRFAbdADat, Ubx = bestRFUbxDat, AbdB = bestRFAbdBDat)

df.m <- melt(df)

pdf("RF_Xval_Boxplots_AUC.pdf")
ggplot(df.m, aes(x=variable, y=value)) + geom_boxplot() + ggtitle("RF 10-fold X-val") + xlab("Gene Model") + ylab("AUC") + theme_classic(base_size=25)
dev.off()

bestRFUbxDat<-as.numeric(read.delim(bestRFUbx, header=F, sep=",")[2,])
bestRFAbdADat<-as.numeric(read.delim(bestRFAbdA, header=F, sep=",")[2,])
bestRFAbdBDat<-as.numeric(read.delim(bestRFAbdB, header=F, sep=",")[2,])


df<-data.frame(AbdA = bestRFAbdADat, Ubx = bestRFUbxDat, AbdB = bestRFAbdBDat)

df.m <- melt(df)

pdf("RF_Xval_Boxplots_Odds.pdf")
ggplot(df.m, aes(x=variable, y=value)) + geom_boxplot() + ggtitle("RF 10-fold X-val") + xlab("Gene Model") + ylab("Odds Ratio") + theme_classic(base_size=25)
dev.off()

