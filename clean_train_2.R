setwd("data_analysis/kaggle/animal/")

library(lubridate)
library(gbm)

read.csv("data/train.csv") -> train
read.csv("data/test.csv") -> test

popularBreeds <- names(summary(train$Breed,maxsum=10L))
trainNameSummary <- summary(train$Name,maxsum=Inf)

clean <- function(x){
    x$Hour <- hour(x$DateTime)
    x$Weekday <- wday(x$DateTime)
    x$DateTime <- as.numeric(as.POSIXct(x$DateTime))
    x$OutcomeSubtype <- NULL
    x$NameLen <- nchar(as.character(x$Name))
    x$NameWeirdness <- trainNameSummary[match(x$Name,names(trainNameSummary))]
    x$NameWeirdness[is.na(x$NameWeirdness)] <- 9999
    x$Name <- NULL
    x$AgeuponOutcome <- gsub(" years?","0000",x$AgeuponOutcome)
    x$AgeuponOutcome <- gsub(" months?","00",x$AgeuponOutcome)
    x$AgeuponOutcome <- gsub(" weeks?","0",x$AgeuponOutcome)
    x$AgeuponOutcome <- gsub(" days?","",x$AgeuponOutcome)
    x$AgeuponOutcome <- as.numeric(paste0("0",x$AgeuponOutcome))
    x$AnimalID <- NULL
    for(i in c("Black","White","Brown","Blue","Orange","Calico","Chocolate","Gold","Red","Tan","Tortie","Yellow")) x[[paste0("col.",i)]] <- grepl(i,x$Color)
    x$Color <- NULL
    for(i in popularBreeds) x[[paste0("breed.",make.names(i))]] <- x$Breed == i
    x$Breed <- NULL
    x
}

train <- clean(train)
test <- clean(test)

for(i in names(train)) {
    if(is.logical(train[[i]])) {
        train[[i]] <- as.numeric(train[[i]])
        test[[i]] <- as.numeric(test[[i]])
    }
}


# write.csv(train, "data/train_clean3.csv",quote = F, row.names = F)
# write.csv(test, "data/test_clean3.csv",quote = F, row.names = F)

###########################################################################################

library(ggplot2)
library(dplyr)
library(xgboost)
library(lubridate)
library(ranger)
library(gbm)
library(Matrix)


train <- read.csv("data/train_clean_31.csv")
test <- read.csv("data/test_clean_31.csv")

set.seed(123)

#model_ranger <- ranger(OutcomeType ~., data = train)
#model_ranger
#ranger(OutcomeType ~., data = train, probability = TRUE, num.trees = 200)

#model_ranger <- ranger(OutcomeType ~., data = train, probability = TRUE, num.trees = 300, mtry = 15)
#ranger(OutcomeType ~., data = train, probability = TRUE, num.trees = 300, mtry = 20)

design_matrix <- model.matrix(OutcomeType ~ .-OutcomeType, data = train)

y <- as.numeric(train$OutcomeType) - 1

test_ID <- test$ID
test$ID <- NULL

# cross validation
xgb.cv(data = design_matrix, label = y, max_depth = 6,
       nrounds = 500,
       eta = 0.05,
       objective = "multi:softprob", 
       num_class = 5, nfold = 5) -> cv

# which wins
which.min(cv$test.merror.mean)

min(cv$test.merror.mean)

# plot error
ggplot(cv, aes(x = 1:500)) +
    geom_line(aes(y = train.merror.mean), alpha = 0.5) +
    geom_line(aes(y = test.merror.mean), color = "red") +
    geom_point(aes(x = which.min(cv$test.merror.mean),
                   y = min(cv$test.merror.mean)), size = 5)+
    ylim(c(0.2, 0.4))

# final train
model_xgb <- xgboost(data = design_matrix, label = y,
                     max_depth = 6,
                     nrounds = 500,
                     eta= 0.05,
                     objective = "multi:softprob",
                     num_class = 5)

# predict
test$NameWeirdness[is.na(test$NameWeirdness)] <- 9999
dtest <- data.matrix(test)
pred <- predict(model_xgb, dtest)
pred_mat <- matrix(pred, nrow = 11456, byrow = F)

# solution
solution <- data.frame("ID" = test_ID, pred_mat)
colnames(solution) <- c("ID","Adoption","Died","Euthanasia","Return_to_owner","Transfer")
write.csv(solution, "submission/model_xgb_clean_data_3.csv", row.names = F, quote = F)
