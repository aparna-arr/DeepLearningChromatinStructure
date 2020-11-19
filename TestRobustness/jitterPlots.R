library(reshape2)
library(ggplot2)

cnn<-read.delim("/oak/stanford/groups/aboettig/Aparna/NNreviews/TestRobustness/ConvNetTrain/meta.txt", sep=',', header=T)
rf<-read.delim("/oak/stanford/groups/aboettig/Aparna/NNreviews/TestRobustness/RFTrain/meta.txt", sep=',', header=T)


dat <- cbind.data.frame(cnn,rf[,2])
colnames(dat) <- c('NoiseDiameter', 'CNN_DevAUC', 'RF_DevAUC')

dat.m <- melt(dat,id.vars=c('NoiseDiameter'))


pdf("JitterComparison.pdf")

ggplot(dat.m, aes(NoiseDiameter,value, color=variable)) + geom_point() + geom_line() + ggtitle("Jitter noise robustness test") + xlab("Noise allowed diameter") + ylab('Dev AUC')

dev.off()

png("JitterComparison.png")

ggplot(dat.m, aes(NoiseDiameter,value, color=variable)) + geom_point() + geom_line() + ggtitle("Jitter noise robustness test") + xlab("Noise allowed diameter") + ylab('Dev AUC')

dev.off()
