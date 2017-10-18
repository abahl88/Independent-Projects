#Data partitioning and modeling
library(outliers)
library(caret)
library(dplyr)

#Setting the working directory
setwd('E:/AML-BUAN 6341')

#Reading in the data
df <- read.csv('Attrition.csv')
colnames(df)[1] <- "Age"


#Dropping categorical variable with just 1 level
df = select(df, -EmployeeCount,-StandardHours,-Over18)

#Converting to categorical variables
 
df$Education <- as.factor(df$Education) 
df$EnvironmentSatisfaction <- as.factor(df$EnvironmentSatisfaction)
df$JobInvolvement <- as.factor(df$JobInvolvement)
df$JobLevel <- as.factor(df$JobLevel)
df$JobSatisfaction <- as.factor(df$JobSatisfaction)
df$PerformanceRating <- as.factor(df$PerformanceRating)
df$RelationshipSatisfaction <- as.factor(df$RelationshipSatisfaction)
df$StockOptionLevel <- as.factor(df$StockOptionLevel)
df$TrainingTimesLastYear <- as.factor(df$TrainingTimesLastYear)
df$WorkLifeBalance <- as.factor(df$WorkLifeBalance)

str(df)

#Scaling Continous Features
library(BBmisc)
nums <- sapply(df, is.numeric)
numeric <- df[,nums]
scalenumeric <- normalize(numeric, method = "standardize", range = c(0, 1))


categ <- sapply(df, is.factor)
categorical <- numeric <- df[,categ]

df2 <- cbind.data.frame(scalenumeric, categorical)

#Dummy variable conversion
X <- select(df2,-Attrition)
dmy <- dummyVars(" ~ . ", data = X ) 
df3 <- data.frame(predict(dmy, newdata = X))
df4 <-  cbind.data.frame(df3, df2$Attrition)
colnames(df4)[86] <- "Attrition"
str(df4)

#1. Support Vector Machines (SVM)

#Partitioning the dataset
trainIndex <- createDataPartition(df4$Attrition , p = 0.7 ,list = FALSE, times = 1)

svm_train_df <- df4[trainIndex,]
svm_test_df <- df4[-trainIndex,]


library(e1071)

svm_tune <- tune(svm, Attrition ~ ., data = svm_train_df, ranges = list(gamma = 10^(-5:-2), cost = 2^(1:5)))
print(svm_tune)

plot(svm_tune , col = 'lightblue')

#- best parameters:
#gamma cost
#0.001    16

#Radial Kernel

model_svm_radial <- svm(Attrition ~ . , svm_train_df  , gamma = 0.001  ,  cost = 16)
summary(model_svm_radial)

radial_svm_pred <- predict(model_svm_radial, svm_test_df)

cm <- table(fitted = radial_svm_pred, actual = svm_test_df$Attrition)
cm

#Plotting Area under curve
library(AUC)

plot(roc(radial_svm_pred,svm_test_df$Attrition))

auc(roc(radial_svm_pred,svm_test_df$Attrition))

#Linear Kernel

model_svm_linear <- svm(Attrition ~ . , svm_train_df , kernel = 'linear', gamma = 0.001  ,  cost = 16 )
summary(model_svm_linear)

linear_svm_pred <- predict(model_svm_linear, svm_test_df)

cm <- table(fitted = linear_svm_pred, actual = svm_test_df$Attrition)
cm

#Plotting Area under curve
plot(roc(linear_svm_pred,svm_test_df$Attrition))

auc(roc(linear_svm_pred,svm_test_df$Attrition))


#Polynomial Kernel

model_svm_poly <- svm(Attrition ~ . , svm_train_df , kernel = 'polynomial', gamma = 0.001  ,  cost = 16, 
                                      coef0 = 1 , degree = 4)
summary(model_svm_poly)

poly_svm_pred <- predict(model_svm_poly, svm_test_df)

cm_svm <- table(fitted = poly_svm_pred, actual = svm_test_df$Attrition)
cm_svm

#Plotting Area under curve

plot(roc(poly_svm_pred,svm_test_df$Attrition))

auc(roc(poly_svm_pred,svm_test_df$Attrition))


##############################################################################################################

#2. Decision trees
library(rpart)

trainIndex <- createDataPartition(df$Attrition , p = 0.7 ,list = FALSE, times = 1)

dt_train_df <- df[trainIndex,]
dt_test_df <- df[-trainIndex,]

model_dt <- rpart(dt_train_df$Attrition ~ ., data = dt_train_df ,method="class")
plotcp(model_dt)
printcp(model_dt)

ptree<- prune(model_dt, cp= model_dt$cptable[which.min(model_dt$cptable[,"xerror"]),"CP"])
printcp(ptree)

library(rattle)
library(rpart.plot)

fancyRpartPlot(ptree,cex=1)


dt_pred <- predict(ptree,dt_test_df, type="class")

cm_tree <- table(fitted = dt_pred, actual = dt_test_df$Attrition)
cm_tree

#Plotting Area under curve

plot(roc(dt_pred,dt_test_df$Attrition))

auc(roc(dt_pred,dt_test_df$Attrition))


##############################################################################################################

#3. Boosting

library(gbm)
library(caret)

trainIndex <- createDataPartition(df$Attrition , p = 0.7 ,list = FALSE, times = 1)


gbm_train_df <- df[trainIndex,]
gbm_train_df$Attrition <- ifelse(gbm_train_df$Attrition == 'Yes', 1,0)
X <- as.data.frame(select(gbm_train_df, -Attrition))
y <- as.factor(gbm_train_df$Attrition)

gbm_test_df <- df[-trainIndex,]
gbm_test_df$Attrition <- ifelse(gbm_test_df$Attrition == 'Yes', 1,0)


gbmGrid <- expand.grid(interaction.depth = (1:5) * 2, n.trees = (1:25)*25, shrinkage = .1, n.minobsinnode = 1)
trainControl <- trainControl(method="cv", number=10)

gbmFit <- train( X , y ,method = "gbm",verbose = FALSE, bag.fraction = 0.5, tuneGrid = gbmGrid , 
                 metric = 'Accuracy', trControl = trainControl)

plot(gbmFit)
summary(gbmFit)

model_gbm <- gbm(gbm_train_df$Attrition ~ ., data = gbm_train_df, dist="adaboost", n.tree = 400 ,shrinkage = 0.1,
                 interaction.depth = 6 )


gbm_pred <- predict(model_gbm ,newdata = gbm_test_df ,type="response",  n.tree = 400 ,interaction.depth = 6 )
gbm_pred <- round(gbm_pred)

cm <- table(fitted = gbm_pred, actual = gbm_test_df$Attrition)
cm



#Plotting Area under curve

plot(roc(gbm_pred,as.factor(gbm_test_df$Attrition)))

auc(roc(gbm_pred,as.factor(gbm_test_df$Attrition)))

