library(ggplot2)
library(reshape2)


## Ubx ##
avgSimFileUbx<-"/oak/stanford/groups/aboettig/Aparna/NNproject/KfoldLogs/AvgClassNoML_Ubx_KfoldXval.log"
bestCNNUbx<-"/oak/stanford/groups/aboettig/Aparna/NNproject/KfoldLogs/modelconv1-0_learn_1e-05_weight_decay_2e-05_minibatch_32_epochs_500_Ubx_KfoldXval.log"
RFUbx<-"/oak/stanford/groups/aboettig/Aparna/NNreviews/RandomForestBest/num-estimators_900_min-sample-split_15_max-depth_None_max-leaf-nodes_3_random-state_0_class-weight_balanced_rf-Ubx_KfoldXval.log"
SVMUbx<-"/oak/stanford/groups/aboettig/Aparna/NNreviews/SVM/svm-Ubx_KfoldXval.log"


## AbdA ##
avgSimFileAbdA<-"/oak/stanford/groups/aboettig/Aparna/NNproject/KfoldLogs/AvgClassNoML_AbdA_KfoldXval.log"
bestCNNAbdA<-"/oak/stanford/groups/aboettig/Aparna/NNproject/KfoldLogs/modelconv1-0_learn_1e-05_weight_decay_2e-05_minibatch_32_epochs_500_AbdA_KfoldXval.log"
RFAbdA<-"/oak/stanford/groups/aboettig/Aparna/NNreviews/RandomForestBest/num-estimators_900_min-sample-split_15_max-depth_None_max-leaf-nodes_3_random-state_0_class-weight_balanced_rf-AbdA_KfoldXval.log"
SVMAbdA<-"/oak/stanford/groups/aboettig/Aparna/NNreviews/SVM/svm-AbdA_KfoldXval.log"


## AbdB ##
avgSimFileAbdB<-"/oak/stanford/groups/aboettig/Aparna/NNproject/KfoldLogs/AvgClassNoML_AbdB_KfoldXval.log"
bestCNNAbdB<-"/oak/stanford/groups/aboettig/Aparna/NNproject/KfoldLogs/modelconv1-0_learn_1e-05_weight_decay_2e-05_minibatch_32_epochs_500_AbdB_KfoldXval.log"
RFAbdB<-"/oak/stanford/groups/aboettig/Aparna/NNreviews/RandomForestBest/num-estimators_900_min-sample-split_15_max-depth_None_max-leaf-nodes_3_random-state_0_class-weight_balanced_rf-AbdB_KfoldXval.log"
SVMAbdB<-"/oak/stanford/groups/aboettig/Aparna/NNreviews/SVM/svm-AbdB_KfoldXval.log"

## Enhancer ##

enhancerDat<-read.delim("/oak/stanford/groups/aboettig/Aparna/NNproject/KfoldLogs/enhancerOddsRatios.csv", header=T, sep=",")

########## CODE ############

avgSimUbxDat<-as.numeric(read.delim(avgSimFileUbx, header=F, sep=",")[2,])
bestCNNUbxDat<-as.numeric(read.delim(bestCNNUbx, header=F, sep=",")[2,])
RFUbxDat<-as.numeric(read.delim(RFUbx, header=F, sep=",")[2,])
SVMUbxDat<-as.numeric(read.delim(SVMUbx, header=F, sep=",")[2,])
avgSimAbdADat<-as.numeric(read.delim(avgSimFileAbdA, header=F, sep=",")[2,])
bestCNNAbdADat<-as.numeric(read.delim(bestCNNAbdA, header=F, sep=",")[2,])
RFAbdADat<-as.numeric(read.delim(RFAbdA, header=F, sep=",")[2,])
SVMAbdADat<-as.numeric(read.delim(SVMAbdA, header=F, sep=",")[2,])
avgSimAbdBDat<-as.numeric(read.delim(avgSimFileAbdB, header=F, sep=",")[2,])
bestCNNAbdBDat<-as.numeric(read.delim(bestCNNAbdB, header=F, sep=",")[2,])
RFAbdBDat<-as.numeric(read.delim(RFAbdB, header=F, sep=",")[2,])
SVMAbdBDat<-as.numeric(read.delim(SVMAbdB, header=F, sep=",")[2,])

enhancerNum = c(seq(1,4), seq(1,3), seq(1,5))
df = cbind.data.frame(enhancerDat, enhancerNum=enhancerNum, Algorithm=rep("BinContact",12))

ubx<-cbind.data.frame(oddsRatio=c(mean(avgSimUbxDat), mean(SVMUbxDat), mean(RFUbxDat), mean(bestCNNUbxDat)),orSigma=c(sd(avgSimUbxDat), sd(SVMUbxDat), sd(RFUbxDat), sd(bestCNNUbxDat)), geneName=c("Ubx", "Ubx", "Ubx", "Ubx"), enhancerNum=c(1,1), Algorithm=c("AvgSimilarity", "SVM", "RandomForest", "CNN"))

abda<-cbind.data.frame(oddsRatio=c(mean(avgSimAbdADat), mean(SVMAbdADat), mean(RFAbdADat), mean(bestCNNAbdADat)),orSigma=c(sd(avgSimAbdADat), sd(SVMAbdADat), sd(RFAbdADat), sd(bestCNNAbdADat)), geneName=c("AbdA", "AbdA", "AbdA", "AbdA"), enhancerNum=c(1,1), Algorithm=c("AvgSimilarity", "SVM", "RandomForest", "CNN"))

abdb<-cbind.data.frame(oddsRatio=c(mean(avgSimAbdBDat), mean(SVMAbdBDat), mean(RFAbdBDat), mean(bestCNNAbdBDat)),orSigma=c(sd(avgSimAbdBDat), sd(SVMAbdBDat), sd(RFAbdBDat), sd(bestCNNAbdBDat)), geneName=c("AbdB", "AbdB", "AbdB", "AbdB"), enhancerNum=c(1,1), Algorithm=c("AvgSimilarity", "SVM", "RandomForest", "CNN"))


df = rbind.data.frame(df, ubx, abda, abdb)
df<-rbind.data.frame(df[1:12,],
	df[13,],df[17,],df[21,], 
	df[14,], df[18,], df[22,], 
	df[15,], df[19,], df[23,], 
	df[16,], df[20,], df[24,])
df<-cbind.data.frame(df,Bars=seq(1,24))
df$Bars<-paste0(df$Bars,"_", df$geneName)
df$Bars<-factor(df$Bars, levels=df$Bars)
#var=paste0(df$geneName, "_E_", df$enhancerNum, "_", df$Algorithm)
#min.df<-cbind.data.frame(oddsRatio=df$oddsRatio, orSigma=df$orSigma, var=var)

#min.df<-cbind.data.frame(oddsRatio=df$oddsRatio, orSigma=df$orSigma, geneName=df$geneName)
#min.df.m<-melt(min.df, id.vars=c("oddsRatio","orSigma"))
#min.df<-cbind.data.frame(oddsRatio=df$oddsRatio, orSigma=df$orSigma, geneName=df$geneName, Bars=factor(seq(1,18)))

pdf("AllBarplots.pdf")

#ggplot(df, aes(geneName, oddsRatio,group=geneName, colour=factor(enhancerNum), fill=Algorithm)) + geom_bar(stat="identity", position=position_dodge()) + geom_errorbar(aes(ymin=oddsRatio-orSigma, ymax=oddsRatio+orSigma), position=position_dodge()) + ggtitle("") 
ggplot(df, aes(Bars, oddsRatio, fill=Algorithm)) + geom_bar(stat="identity") + geom_errorbar(aes(ymin=oddsRatio-orSigma, ymax=oddsRatio+orSigma)) + ggtitle("")  

dev.off()
