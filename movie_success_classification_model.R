# Setting working directory
setwd("G:/MSBAPM/Career development/Screened applications/Quantiphi")

# Load packages
library('readxl') # Reading excel files
library('ggplot2') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('randomForest') # classification algorithm
library('mltools')  # For generating CV folds and one-hot-encoding
library('data.table')
library('LiblineaR')  # Support Vector Machine and Logistic Regression

#**********************************************PREPROCESSING****************************************************

#Reading the files
train<-read_excel('Training Sheet.xlsx')
train$source=c("train")

test<-read_excel('Scoring Sheet.xlsx')
test$source=c("test")

#Identify variables that are common between two datasets
common_cols <- intersect(colnames(train), colnames(test))

# bind training & test data
full <- rbind(
  subset(train, select = common_cols), 
  subset(test, select = common_cols)
)

str(full)

rm(common_cols)

#write.csv(full,'append.csv')

#Creating a function to calculate number of missing values, proportion of missing values 
#and number of unique values across each column
missing_values = function(input)
{
  n = length(colnames(input)) # number of columns
  a <- NULL
  b <- NULL
  c <- NULL
  for(i in 1:n) 
  {
    a[i]=sum(is.na(input[,i])) 
    b=a/nrow(input) 
    c[i]=nrow(unique(input[,i]))
  }
  result=data.frame(colnames(input),a,b,c) 
  colnames(result) = c("column Name", "# Missing Values", "% Missing Value", "Unique Values")
  return(result) 
}

missing_values(full)

blank_values = function(input)
{
  n = length(colnames(input)) # number of columns
  a <- NULL
  b <- NULL 
  c <- NULL
  for(i in 1:n) 
  {
    a[i]=sum(ifelse(input[,i]=="",1,0)) 
    b=a/nrow(input) 
    c[i]=nrow(unique(input[,i])) 
  }
  result=data.frame(colnames(input),a,b,c) 
  colnames(result) = c("column Name", "# Blank Values", "% Blank Value", "Unique Values")
  return(result) 
}

blank_values(full)

#INFERENCE: NO MISSING VALUES

rm(blank_values,missing_values)


#*********************************************EXPLORATORY ANALYSIS*************************************************
#Frequency distribution of response variable
theme_set(theme_classic())

# Plot frequency of target variable
df <- as.data.frame.table(table(train$Category))
g <- ggplot(df, aes(Var1, Freq))
g + geom_bar(stat="identity", width = 0.5, fill="navy") + 
  labs(title="Frequency of Movie Categories",
       caption="Source: Quantiphi Labs") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

#Observation: Higher the rating, fewer the movies

# Plot avg. category across Creative type
df<-aggregate(train$Category,list(train$creative_type),mean)
g <- ggplot(df, aes(Group.1, x))
g + geom_bar(stat="identity", width = 0.5, fill="navy") + 
  labs(title="Avg. category for creative type",
       caption="Source: Quantiphi Labs") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

#Observation: No two creative types have a similar rating

# Plot avg. category across Movie board rating
df<-aggregate(train$Category,list(train$creative_type),mean)
g <- ggplot(df, aes(Group.1, x))
g + geom_bar(stat="identity", width = 0.5, fill="navy") + 
  labs(title="Avg. category for creative type",
       caption="Source: Quantiphi Labs") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

#Observation:  Movies that are open to general audience have higher success

# Plot avg. category across Production Methods
df<-aggregate(train$Category,list(train$production_method),mean)
g <- ggplot(df, aes(Group.1, x))
g + geom_bar(stat="identity", width = 0.5, fill="navy") + 
  labs(title="Avg. category for production method",
       caption="Source: Quantiphi Labs") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

#Observation: Different types of Action seem to drive success

# Plot avg. category across languages
df<-aggregate(train$Category,list(train$language),mean)
g <- ggplot(df, aes(Group.1, x))
g + geom_bar(stat="identity", width = 0.5, fill="navy") + 
  labs(title="Avg. category for language",
       caption="Source: Quantiphi Labs") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

df<-aggregate(train$Category,list(train$language),length)
g <- ggplot(df, aes(Group.1, x))
g + geom_bar(stat="identity", width = 0.5, fill="navy") + 
  labs(title="# Films for language",
       caption="Source: Quantiphi Labs") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

#Observation: Aggregated comparison of English with other language due to the difference in number of films made  


#Eliminating unnecessary variables: name, display_name,board_rating_reason because there are too many classes

full2=full[,-c(2,3,11)]

full2$production_year=as.character(full2$production_year)

#Dummifying categorical variables
library(caret)
dmy <- dummyVars(" ~ .", data = full2)
trsf <- data.frame(predict(dmy, newdata = full2))
View(trsf)


library("dplyr")
train_featurized=left_join(train[,c(1,15)],trsf,by=c("id"))
test_featurized=left_join(test[,c(1)],trsf,by=c("id"))
#**********************************************MODEL 1: RANDOM FOREST*******************************************


set.seed(1)
train_index = sample(1: nrow(train_featurized), 0.75*nrow(train_featurized))
attach(train_featurized)

rf_fit =randomForest(factor(Category)~.,data=train_featurized[,-c(1,19,20)],subset =train_index ,
                           mtry=8, importance =TRUE)

rf_fit
summary(rf_fit)

validate_rf=predict(rf_fit,newdata =train_featurized[-train_index,-c(1,19,20)])
validation=train_featurized[-train_index,2]
validation$Category=as.factor(validation$Category)

#Accuracy on Validation dataset
validate_rf=as.data.frame.character(validate_rf)
accuracy_validation=cbind(validate_rf,validation$Category)
sum(ifelse(accuracy_validation$validate_rf==accuracy_validation$`validation$Category`,0,1))/299

# Get importance
importance    <- importance(rf_fit)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

rm(accuracy_validation,validate_rf,importance,rankImportance,dmy,trsf)

#*************************MODEL 2: RANDOM FOREST WITH REDUCED CLASSES FOR CATEGORICAL VARIABLES****************

train_featurized2=train_featurized

#Add less important classes from Model 1 in Creative type
train_featurized2$other_creatives=apply(train_featurized[,c(11,12,14,15,16,18)],1,sum)

#Drop individual classes in Variable creative type
train_featurized2=train_featurized2[,-c(11,12,14,15,16,18)]

#Add production methods other than Non-Action to a single variable
train_featurized2$production_non_action=apply(train_featurized2[,c(15,16,17,19,20)],1,sum)
train_featurized2=train_featurized2[,-c(15,16,17,19,20)]

#Convert language variables to English vs Non-english
train_featurized2$language_nonenglish=apply(train_featurized2[,c(29:45)],1,sum)
train_featurized2$language_nonenglish=(train_featurized2$language_nonenglish)-(train_featurized2$languageEnglish)

train_featurized2=train_featurized2[,-c(29:45)]

## Making MPAA ratings into three levels based on the insights from secondary research
#https://www.researchgate.net/publication/305800376_Do_MPAA_Ratings_Affect_Box_Office_Revenues

train_featurized2$boardrating_medium=apply(train_featurized2[,c(32:34)],1,sum)
train_featurized2=train_featurized2[,-c(32:34)]

attach(train_featurized2)

set.seed(1)
train_index2 = sample(1: nrow(train_featurized), 0.70*nrow(train_featurized))

rf_fit2 =randomForest(factor(Category)~.,data=train_featurized2[,-c(1,13,14)],subset =train_index2,
                     mtry=6, importance =TRUE)

rf_fit2
summary(rf_fit2)

validate_rf2=predict(rf_fit2,newdata =train_featurized2[-train_index2,-c(1,19,20)])
validation2=train_featurized2[-train_index2,2]
validation2$Category=as.factor(validation2$Category)

validate_rf2=as.data.frame.character(validate_rf2)
accuracy_validation2=cbind(validate_rf2,validation2$Category)


# Get importance
importance2    <- importance(rf_fit2)
varImportance2 <- data.frame(Variables = row.names(importance2), 
                            Importance = round(importance2[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance2 <- varImportance2 %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

rm(importance2,rankImportance2)


#********************************MODEL 3: CROSS VALIDATED SVM WITH FEWER VARIABLES*****************************

#identify zero variance variables

col_variance=data.frame(apply(train_featurized2,2,sd))

#Removing variables with less importance from previous model and those with zero variance
train_featurized3=train_featurized2[,-c(8,13,14,20,34,36,37)]

train_featurized3=data.table(train_featurized3)
train_featurized3[, FoldID := folds(Category, nfolds=5, stratified=TRUE, seed=1234)]  # mltools function


svmCV <- list()
svmCV[["Features"]] <- colnames(train_featurized3[,c(3:36)])
svmCV[["ParamGrid"]] <- CJ(type=1:5, cost=c(.01, .1, 1, 10,100), Score=NA_real_)
svmCV[["BestScore"]] <- 0

# Loop through each set of parameters
for(i in seq_len(nrow(svmCV[["ParamGrid"]]))){
  
  # Get the ith set of parameters
  params <- svmCV[["ParamGrid"]][i]
  
  # Build an empty vector to store scores from each train/test fold
  scores <- numeric()
  
  # Build an empty list to store predictions from each train/test fold
  predsList <- list()
  
  # Loop through each test fold, fit model to training folds and make predictions on test fold
  for(foldID in 1:5){
    
    # Build the train/test folds
    testFold <- train_featurized3[J(FoldID=foldID), on="FoldID"]
    trainFolds <- train_featurized3[!J(FoldID=foldID), on="FoldID"]  # Exclude fold i from trainFolds
    
    # Train the model & make predictions
    svm <- LiblineaR(data=trainFolds[, svmCV$Features, with=FALSE], target=trainFolds$Category, type=params$type, cost=params$cost)
    testFold[, Pred := predict(svm, testFold[, svmCV$Features, with=FALSE])$predictions]
    predsList <- c(predsList, list(testFold[, list(id, FoldID, Pred)]))
    
    # Evaluate predictions (accuracy rate) and append score to scores vector
    score <- mean(testFold$Pred == testFold$Category)
    scores <- c(scores, score)
  }
  
  # Measure the overall score. If best, tell svmCV
  score <- mean(scores)
  
  # Insert the score into ParamGrid
  svmCV[["ParamGrid"]][i, Score := score][]
  print(paste("Params:", paste(colnames(svmCV[["ParamGrid"]][i]), svmCV[["ParamGrid"]][i], collapse = " | ")))
  
  if(score > svmCV[["BestScore"]]){
    svmCV[["BestScores"]] <- scores
    svmCV[["BestScore"]] <- score
    svmCV[["BestParams"]] <- svmCV[["ParamGrid"]][i]
    svmCV[["BestPreds"]] <- rbindlist(predsList)
  }
}

# Check the best parameters
svmCV[["BestParams"]]

# Extract predictions
metas.svm <- svmCV[["BestPreds"]]




#********************************MODEL 4: CROSS VALIDATED KNN WITH FEWER VARIABLES*****************************

#
# Do a grid search for k = 1, 2, ... 30 by cross validating model using folds 1-5
# I.e. [test=f1, train=(f2, f3, f4, f5)], [test=f2, train=(f1, f3, f4, f5)], ...
library(class)

knnCV <- list()
knnCV[["Features"]] <- colnames(train_featurized3[,c(3:36)])
knnCV[["ParamGrid"]] <- CJ(k=seq(1, 30))
knnCV[["BestScore"]] <- 0


# Loop through each set of parameters
for(i in seq_len(nrow(knnCV[["ParamGrid"]]))){
  
  # Get the ith set of parameters
  params <- knnCV[["ParamGrid"]][i]
  
  # Build an empty vector to store scores from each train/test fold
  scores <- numeric()
  
  # Build an empty list to store predictions from each train/test fold
  predsList <- list()
  
  # Loop through each test fold, fit model to training folds and make predictions on test fold
  for(foldID in 1:5){
    
    # Build the train/test folds
    testFold <- train_featurized3[J(FoldID=foldID), on="FoldID"]
    trainFolds <- train_featurized3[!J(FoldID=foldID), on="FoldID"]  # Exclude fold i from trainFolds
    
    # Train the model & make predictions
    testFold[, Pred := knn(train=trainFolds[, knnCV$Features, with=FALSE], test=testFold[, knnCV$Features, with=FALSE], cl=trainFolds$Category, k=params$k)]
    predsList <- c(predsList, list(testFold[, list(id, FoldID, Pred)]))
    
    # Evaluate predictions (accuracy rate) and append score to scores vector
    score <- mean(testFold$Pred == testFold$Category)
    scores <- c(scores, score)
  }
  
  # Measure the overall score. If best, tell knnCV
  score <- mean(scores)
  
  # Insert the score into ParamGrid
  knnCV[["ParamGrid"]][i, Score := score][]
  print(paste("Params:", paste(colnames(knnCV[["ParamGrid"]][i]), knnCV[["ParamGrid"]][i], collapse = " | ")))
  
  if(score > knnCV[["BestScore"]]){
    knnCV[["BestScores"]] <- scores
    knnCV[["BestScore"]] <- score
    knnCV[["BestParams"]] <- knnCV[["ParamGrid"]][i]
    knnCV[["BestPreds"]] <- rbindlist(predsList)
  }
}

# Check the best parameters
knnCV[["BestParams"]]


# Plot the score for each k value
knnCV[["ParamGrid"]]
ggplot(knnCV[["ParamGrid"]], aes(x=k, y=Score))+geom_line()+geom_point()


# Extract predictions
metas.knn <- knnCV[["BestPreds"]]
metas.svm <- svmCV[["BestPreds"]]


# Insert regular predictions into train
train_featurized3[metas.knn, Meta.knn := Pred, on="id"]
train_featurized3[metas.svm, Meta.svm := Pred, on="id"]


test_featurized3=data.table(test_featurized3)
train_featurized3=data.table(train_featurized3)

#************************************************Test features***************************************************

test_featurized2=test_featurized
#Add less important classes from Model 1 in Creative type
test_featurized2$other_creatives=apply(test_featurized[,c(10,11,13,14,15,17)],1,sum)

#Drop individual classes in Variable creative type
test_featurized2=test_featurized2[,-c(10,11,13,14,15,17)]

#Add production methods other than Non-Action to a single variable
test_featurized2$production_non_action=apply(test_featurized2[,c(14,15,16,18,19)],1,sum)
test_featurized2=test_featurized2[,-c(14,15,16,18,19)]

#Convert language variables to English vs Non-english
test_featurized2$language_nonenglish=apply(test_featurized2[,c(28:44)],1,sum)
test_featurized2$language_nonenglish=(test_featurized2$language_nonenglish)-(test_featurized2$languageEnglish)

test_featurized2=test_featurized2[,-c(28:44)]

## Making MPAA ratings into three levels based on the insights from secondary research
test_featurized2$boardrating_medium=apply(test_featurized2[,c(31:33)],1,sum)
test_featurized2=test_featurized2[,-c(31:33)]

test_featurized3=test_featurized2[,intersect(colnames(test_featurized2),colnames(train_featurized3))]

#******************************************FINAL MODEL WITH PREDICTIONS FROM KNN**********************
test_featurized3[, Pred := knn(train=train_featurized3[, knnCV$Features[-34], with=FALSE], test=test_featurized3[, knnCV$Features[-34], with=FALSE], cl=train_featurized3$Category, k=7)]


scoring_sheet_submission=left_join(test,test_featurized3,by=c("id"))


write.csv(scoring_sheet_Yashwanth,'scoring_sheet_Yashwanth.csv')