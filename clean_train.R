setwd("data_analysis/kaggle/animal/")

library(lubridate)
library(stringr)
library(xgboost)

train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

train$AnimalID <- NULL
train$OutcomeSubtype <- NULL
test_ID <- test$ID
test$ID <- NULL

# Add some date/time-related variables

train$DateTime <- as.POSIXct(train$DateTime)
test$DateTime <- as.POSIXct(test$DateTime)

train$year <- year(train$DateTime)
train$month <- month(train$DateTime)
train$wday <- wday(train$DateTime)
train$hour <- hour(train$DateTime)

test$year <- year(test$DateTime)
test$month <- month(test$DateTime)
test$wday <- wday(test$DateTime)
test$hour <- hour(test$DateTime)

train$DateTime <- as.numeric(train$DateTime)
test$DateTime <- as.numeric(test$DateTime)

# Write a function to convert age outcome to numeric age in days
convert <- function(age_outcome){
    split <- strsplit(as.character(age_outcome), split=" ")
    period <- split[[1]][2]
    if (grepl("year", period)){
        per_mod <- 365
    } else if (grepl("month", period)){ 
        per_mod <- 30
    } else if (grepl("week", period)){
        per_mod <- 7
    } else
        per_mod <- 1
    age <- as.numeric(split[[1]][1]) * per_mod
    return(age)
}

train$AgeuponOutcome <- sapply(train$AgeuponOutcome, FUN=convert)
test$AgeuponOutcome <- sapply(test$AgeuponOutcome, FUN=convert)
train[is.na(train)] <- 0  # Fill NA with 0
test[is.na(test)] <- 0

# Remove row with missing sex label and drop the level
train <- train[-which(train$SexuponOutcome == ""),]
train$SexuponOutcome <- droplevels(train$SexuponOutcome)

# Add var for name length
train$name_len <- sapply(as.character(train$Name),nchar)
test$name_len <- sapply(as.character(test$Name),nchar)

train$Name <- NULL
test$Name <- NULL

# Create indicator vars for breeds and mix
train_breeds <- as.character(train$Breed)
test_breeds <- as.character(test$Breed)
all_breeds <- unique(c(train_breeds,test_breeds))
breed_words <- unique(unlist(strsplit(all_breeds, c("/| Mi")))) 

for (breed in breed_words){
    train[breed] <- as.numeric(grepl(breed, train_breeds))
    test[breed] <- as.numeric(grepl(breed, test_breeds))
}

train["crosses"] <- str_count(train$Breed, pattern="/")
test["crosses"] <- str_count(test$Breed, pattern="/")

train$Breed <- NULL
test$Breed <- NULL

# Create indicator vars for color
train_colors <- as.character(train$Color)
test_colors <- as.character(test$Color)
all_colors <- unique(c(train_colors,test_colors))
color_words <- unique(unlist(strsplit(all_colors, c("/")))) 

for (color in color_words){
    train[color] <- as.numeric(grepl(color, train_colors))
    test[color] <- as.numeric(grepl(color, test_colors))
}

train["color_count"] <- str_count(train$Color, pattern="/")+1
test["color_count"] <- str_count(test$Color, pattern="/")+1

train$Color <- NULL
test$Color <- NULL

targets <- train$OutcomeType
train$OutcomeType <- NULL

# Submission code
set.seed(213)
full_train_matrix <- matrix(as.numeric(data.matrix(train)),ncol=306)
test_matrix <- matrix(as.numeric(data.matrix(test)),ncol=306)

full_targets_train <- as.numeric(targets)-1

# Run xgb on full train set
xgb_model_test = xgboost(data=full_train_matrix, 
                         label=full_targets_train, 
                         nrounds=125, 
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

write.csv(submission , "submission/model_xgb_r.csv", row.names=FALSE)

