library(ggplot2)
library(DataExplorer)
library(lattice)
library(data.table)
library(mltools)
library(tidyverse)
library(caret)
library(reshape2)
library(ISLR)
library(glmnet)
library(MXM)
library(lars)
library(pls)
library(timeDate)
library(leaps)

set.seed(420)
df_norm = data.frame(matrix(rnorm(20000), nrow=1000))
B_vector = runif(n = 20, min = -10, max = 50)
B_vector[c(2,4,6,8,10,12,14)] = 0
error_vector = runif(n = 20, min = -2, max = 2)


Y = as.matrix(df_norm) %*% B_vector + error_vector

df_final = cbind(df_norm,Y)
training_samples = df_final$Y %>% createDataPartition(p = 0.1, list = FALSE)
train_data = df_final[training_samples, ]
test_data = df_final[-training_samples, ]

models = regsubsets(Y~., data = train_data, nvmax = 20)
model_summary = summary(models)
model_summary$adjr2
plot(models,scale="bic")
which.min(model_summary$rss)
plot(model_summary$rss ,xlab="Number of Variables ",ylab="RSS",type='l')
points(which.min(model_summary$rss), model_summary$rss[which.min(model_summary$rss)],
       col = "red", cex = 2, pch = 20)
#test data modelling

# Initialize a vector to contain all test errors

test_errors=rep(NA,20)

test_data_matrix= model.matrix(Y~.,data=test_data)

for(i in 1:20){
  coefi=coef(models,id=i)
  pred=test_data_matrix[,names(coefi)]%*%coefi
  test_errors[i]=mean((test_data$Y-pred)^2)
}

train_errors=rep(NA,20)

train_data_matrix= model.matrix(Y~.,data=train_data)

for(i in 1:20){
  coefi=coef(models,id=i)
  pred=train_data_matrix[,names(coefi)]%*%coefi
  train_errors[i]=mean((train_data$Y-pred)^2)
}


#quartz()
plot(test_errors,ylab="MSE",xlab = "Number of Variables", main = "MSE vs Number of variables in train and test data",
pch=20,type="b")
points(train_errors,col="blue",pch=20,type="b")
points(which.min(test_errors), test_errors[which.min(test_errors)],
       col = "red", cex = 2, pch = 20)
points(which.min(train_errors), train_errors[which.min(train_errors)],
       col = "green", cex = 2, pch = 20)
legend("topright",legend=c("Training","Test"),col=c("blue","black"),pch=20)


which.min(train_errors)
which.min(test_errors)

coef(models, id = which.min(test_errors))
coef(models, id=3)
B_vector
