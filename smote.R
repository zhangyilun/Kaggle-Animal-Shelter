# SMOTE to over sample

# set wd
setwd("data_analysis/kaggle/animal/")

# library
library(DMwR)

# load data
train <- read.csv("data/train_clean.csv")
table(train$OutcomeType)

# test <- read.csv("data/test_clean.csv")

# smote
newData <- SMOTE(OutcomeType ~ ., 
                 data = train, 
                 perc.over = 6000, 
                 perc.under = 1000)

# check
table(newData$OutcomeType)

# save
write.csv(newData,"data/train_clean_smote_2.csv",row.names = F, quote = F)
# write.csv(test,"data/test_clean.csv",row.names = F, quote = F)
