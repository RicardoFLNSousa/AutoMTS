suppressMessages(require("reticulate"))
#library("mvnmle")
#library("imp4p")
#install.packages("mtsdi")
#library("mtsdi")
#library("mnimput")
#install.packages("CoImp")
#library("CoImp")

options(warn=-1)

args <- commandArgs(trailingOnly = TRUE)
myData <- read.csv("input_data.csv", sep = ',')
newData <- NULL

print("R PRINTS-------------------------")
print(args)

if(args[1] == 'mice'){
	suppressMessages(require("mice", warn.conflicts = FALSE))
	tempData <- mice(myData, m = strtoi(args[2], base = 0L), method = args[3], maxit = strtoi(args[4], base = 0L) , seed = strtoi(tail(args, n=1), base = 0L)) #predictive mean matching being used pmm as default
 	newData <- complete(tempData, 1)
}else if(args[1] == 'amelia'){
	suppressMessages(require("Amelia"))
 	tempData <- amelia(myData, m = strtoi(args[2], base = 0L), autopri = as.numeric(args[3]), max.resample = strtoi(args[4], base = 0L), p2s=0)
 	newData <- tempData$imputations[1]
}else{
 	newData <- myData
}


write.csv(newData, file = "output_data.csv")
