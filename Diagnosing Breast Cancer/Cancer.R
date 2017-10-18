#Data partitioning and modeling
library(caret)
library(dplyr)

#Setting the working directory
setwd('E:/AML-BUAN 6341')

#Reading in the data
df <- read.csv('cancer.csv')
df <- select(df , -X)

#Scaling Continous Features
library(BBmisc)
nums <- sapply(df, is.numeric)
numeric <- df[,nums]
df_scale <- normalize(numeric, method = "standardize", range = c(0, 1))

#Consolidating the dataset

df1 <- cbind.data.frame(df_scale , df$diagnosis)

#Dropping ID feature
df1 <- select(df1 , -id)

#renaming target variable

colnames(df1)[31] <- "diagnosis"


#1. Support Vector Machines (SVM)


#Partitioning the dataset
trainIndex <- createDataPartition(df1$diagnosis , p = 0.7 ,list = FALSE, times = 1)

svm_train_df <- df1[trainIndex,]
svm_test_df <- df1[-trainIndex,]

library(e1071)

svm_tune <- tune(svm, diagnosis ~ ., data = svm_train_df, ranges = list(gamma = 10^(-5:-2), cost = 2^(1:5)))
print(svm_tune)

plot(svm_tune , col = 'lightblue')

#- best parameters:
#  gamma cost
#0.01    4


#Performance comparisons  -kernels in SVM

#Radial Kernel

model_svm_radial <- svm(diagnosis ~ . , svm_train_df  , gamma = 0.01  ,  cost = 4)
summary(model_svm_radial)

radial_svm_pred <- predict(model_svm_radial, svm_test_df)

cm <- table(fitted = radial_svm_pred, actual = svm_test_df$diagnosis)
cm

#Plotting Area under curve
library(AUC)

plot(roc(radial_svm_pred,svm_test_df$diagnosis))

auc(roc(radial_svm_pred,svm_test_df$diagnosis))

#Linear Kernel

model_svm_linear <- svm(diagnosis ~ . , svm_train_df , kernel = 'linear', gamma = 0.01  ,  cost = 4 )
summary(model_svm_linear)

linear_svm_pred <- predict(model_svm_linear, svm_test_df)

cm <- table(fitted = linear_svm_pred, actual = svm_test_df$diagnosis)
cm

#Plotting Area under curve
plot(roc(linear_svm_pred,svm_test_df$diagnosis))

auc(roc(linear_svm_pred,svm_test_df$diagnosis))


#Polynomial Kernel

model_svm_poly <- svm(diagnosis ~ . , svm_train_df , kernel = 'polynomial', gamma = 0.001  ,  cost = 8, 
                      coef0 = 1 , degree = 4)
summary(model_svm_poly)

poly_svm_pred <- predict(model_svm_poly, svm_test_df)

cm_svm <- table(fitted = poly_svm_pred, actual = svm_test_df$diagnosis)
cm_svm

#Plotting Area under curve

plot(roc(poly_svm_pred,svm_test_df$diagnosis))

auc(roc(poly_svm_pred,svm_test_df$diagnosis))

##############################################################################################################

#2. Decision trees

library(rpart)

trainIndex <- createDataPartition(df$diagnosis , p = 0.7 ,list = FALSE, times = 1)

dt_train_df <- df[trainIndex,]
dt_test_df <- df[-trainIndex,]

model_dt <- rpart(dt_train_df$diagnosis ~ ., data = dt_train_df ,method="class")
plotcp(model_dt)
printcp(model_dt)

ptree<- prune(model_dt, cp= model_dt$cptable[which.min(model_dt$cptable[,"xerror"]),"CP"])

library(rattle)
library(rpart.plot)

fancyRpartPlot(model_dt,cex=1)


dt_pred <- predict(ptree,dt_test_df,type="class")
dt_pred


cm <- table(fitted = dt_pred, actual = dt_test_df$diagnosis)
cm


#Plotting Area under curve

plot(roc(dt_pred,dt_test_df$diagnosis))

auc(roc(dt_pred,dt_test_df$diagnosis))


##############################################################################################################

#Boosting

library(gbm)
library(caret)

trainIndex <- createDataPartition(df$diagnosis , p = 0.7 ,list = FALSE, times = 1)

gbm_train_df <- df[trainIndex,]
gbm_train_df$diagnosis <- ifelse(gbm_train_df$diagnosis == 'M', 1,0)
X <- as.data.frame(select(gbm_train_df, -diagnosis))
y <- as.factor(gbm_train_df$diagnosis)

gbm_test_df <- df[-trainIndex,]
gbm_test_df$diagnosis <- ifelse(gbm_test_df$diagnosis == 'M', 1,0)

gbmGrid <- expand.grid(interaction.depth = (1:5) * 2, n.trees = (1:20)*25, shrinkage = .1, n.minobsinnode = 1)
trainControl <- trainControl(method="cv", number=10)

gbmFit <- train( X , y ,method = "gbm",verbose = FALSE, bag.fraction = 0.5, tuneGrid = gbmGrid , 
                 metric = 'Accuracy', trControl = trainControl)

plot(gbmFit)
summary(gbmFit)

model_gbm <- gbm(gbm_train_df$diagnosis ~ ., data = gbm_train_df, dist="adaboost", n.tree = 250 ,shrinkage = 0.1,
                 interaction.depth = 10 )


summary(model_gbm)



gbm_pred <- predict (model_gbm ,newdata = gbm_test_df ,n.tree = 500, type="response")
gbm_pred <- round(gbm_pred)

cm <- table(fitted = gbm_pred, actual = gbm_test_df$diagnosis)
cm

#Plotting Area under curve

plot(roc(gbm_pred,as.factor(gbm_test_df$diagnosis)))

auc(roc(gbm_pred,as.factor(gbm_test_df$diagnosis)))

