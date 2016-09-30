setwd("data_analysis/kaggle/animal/")

library(xgboost)

# load data
train <- read.csv("data/train_clean_2.csv")
test <- read.csv("data/test_clean_2.csv")

# some split
train_Y <- train$OutcomeType
train_X <- train
train_X$OutcomeType <- NULL
test_ID <- test$ID
test$ID <- NULL
test_X <- test

set.seed(123)
full_train_matrix <- matrix(as.numeric(data.matrix(train_X)),ncol=279)
test_matrix <- matrix(as.numeric(data.matrix(test)),ncol=279)
full_targets_train <- as.numeric(train_Y)-1

# cv
xgb_cv <- xgb.cv(data = full_train_matrix,
                 label = full_targets_train,
                 nfold = 5,
                 nrounds = 50,
                 objective = "multi:softprob",
                 eval_metric = "mlogloss",
                 num_class = 5,
                 max_depth = 15,
                 eta = 0.1, 
                 gamma = 0.01)

# plot
plot(xgb_cv$train.mlogloss.mean,type = "l")
lines(xgb_cv$test.mlogloss.mean,col="red")
legend( x="topright", 
               legend=c("train_mlogloss","test_mlogloss"),
               col=c("black","red"), lwd=1, lty=c(1,2))

# model
xgb_model_test <-  xgboost(data=full_train_matrix, 
                           label=full_targets_train, 
                           nrounds=150, 
                           verbose=1, 
                           eta=0.2, 
                           max_depth=6, 
                           subsample=0.75, 
                           colsample_bytree=0.85,
                           objective="multi:softprob", 
                           eval_metric="mlogloss",
                           num_class=5)

test_preds <- predict(xgb_model_test, test_matrix)
test_preds_frame <- data.frame(matrix(test_preds, ncol = 5, byrow=TRUE))
colnames(test_preds_frame) <- levels(targets)

submission <- cbind(data.frame(ID=test_ID), test_preds_frame)

write.csv(submission , "submission/model_xgb_r_mine.csv", row.names=FALSE)
