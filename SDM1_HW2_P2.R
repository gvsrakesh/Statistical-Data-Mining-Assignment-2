library(ggplot2)
library(lattice)
library(DataExplorer)
library(data.table)
library(mltools)
library(tidyverse)
library(reshape2)
library(glmnet)
library(gmodels)
library(MASS)
library(leaps)

getwd()
setwd("/Users/rakeshguduru/Documents/UB Grad stuff/SDM 1/hw2/vguduru")
df = read.table("ticdata2000.txt", header = FALSE)
colnames(df)
names(df)[86] = "output_variable"
df_1 =  df
summary(df$output_variable)
(sum(df$output_variable)/nrow(df))*100

#Running linear regression
model_lm = lm(output_variable~., data = df)
model_lm_1  = lm(output_variable~1, data = df)

#predictions on train data
predictions_lm_train <- model_lm %>% predict(df)
df$output_predict = predictions_lm_train
summary(predictions_lm_train)
df$output_predict = ifelse(df$output_predict<.5,0,1)
df$output_predict = as.factor(df$output_predict)
levels(df$output_predict)
train_lm_table = CrossTable(x=df$output_variable, y = df$output_predict, prop.chisq = FALSE)
train_lm_table_error = train_lm_table$prop.tbl[1,2] + train_lm_table$prop.tbl[2,1]
train_lm_table_precision = train_lm_table$prop.tbl[2,2]/(train_lm_table$prop.tbl[2,2]+train_lm_table$prop.tbl[1,2])
train_lm_table_recall = train_lm_table$prop.tbl[2,2]/(train_lm_table$prop.tbl[2,2]+train_lm_table$prop.tbl[2,1])

#predictions on test data

df_test = read.table("ticeval2000.txt", header = FALSE)
df_test_target = read.table("tictgts2000.txt", header = FALSE)
names(df_test_target)[1] = "output_test"
df_test = cbind(df_test,df_test_target)
df_test$output_test = as.factor(df_test$output_test)

predictions_test = model_lm %>% predict(df_test)
df_test$output_predict = predictions_test
summary(predictions_test)
df_test$output_predict = ifelse(df_test$output_predict<.5,0,1)
df_test$output_predict = as.factor(df_test$output_predict)
levels(df_test$output_predict)
test_lm_table = CrossTable(x=df_test$output_test, y = df_test$output_predict, prop.chisq = FALSE)
test_lm_table_error = test_lm_table$prop.tbl[1,2] + test_lm_table$prop.tbl[2,1]
test_lm_table_precision = test_lm_table$prop.tbl[2,2]/(test_lm_table$prop.tbl[1,2]+test_lm_table$prop.tbl[2,2])
test_lm_table_recall = test_lm_table$prop.tbl[2,2]/(test_lm_table$prop.tbl[2,1]+test_lm_table$prop.tbl[2,2])

#Stepwise Regression Forward
step_model_forward_AIC = stepAIC(model_lm, direction = "forward", trace = TRUE, scope = list(upper = model_lm, lower = model_lm_1))
step_model_forward = regsubsets(output_variable~., data = df_1, method = "forward", nvmax = 85)
forward_summary = summary(step_model_forward)
which.max(forward_summary$adjr2)
coef(step_model_forward, id = which.max(forward_summary$adjr2))
plot(forward_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2", type = "b")

test_forward_precision=rep(NA,85)
test_forward_recall=rep(NA,85)
test_forward_error=rep(NA,85)

test_data_matrix= model.matrix(~.,data=df_test)

for(i in 2:85){
  coefi=coef(step_model_forward,id=2)
  pred=test_data_matrix[,names(coefi)]%*%coefi
  pred[pred>0.5] = 1
  pred[pred<0.5] = 0
  step_forward_test_lm_table = CrossTable(x=df_test$output_test, y = pred, prop.chisq = FALSE)
  test_forward_error[i] = step_forward_test_lm_table$prop.tbl[1,2] + step_forward_test_lm_table$prop.tbl[2,1]
  test_forward_precision[i] = step_forward_test_lm_table$prop.tbl[2,2]/(step_forward_test_lm_table$prop.tbl[2,2]+step_forward_test_lm_table$prop.tbl[1,2])
  test_forward_recall[i] = step_forward_test_lm_table$prop.tbl[2,2]/(step_forward_test_lm_table$prop.tbl[2,2]+step_forward_test_lm_table$prop.tbl[2,1])
  }

test_forward_precision[is.na(test_forward_precision)] = 0
test_forward_recall[is.na(test_forward_recall)] = 0
test_forward_error[is.na(test_forward_error)] = 100

step_forward_test_lm_table_error = min(test_forward_error)
step_forward_test_lm_table_precision = max(test_forward_precision)
step_forward_test_lm_table_recall = max(test_forward_recall)

#Stepwise Regression Backward
step_model_backward = stepAIC(model_lm, direction = "backward", trace = FALSE)
summary(step_model_backward)
predictions_test = step_model_backward %>% predict(df_test)
df_test$output_predict_step_backward = predictions_test
summary(predictions_test)
df_test$output_predict_step_backward = ifelse(df_test$output_predict_step_backward<.5,0,1)
df_test$output_predict_step_backward = as.factor(df_test$output_predict_step_backward)
levels(df_test$output_predict_step_backward)
step_backward_test_lm_table = CrossTable(x=df_test$output_test, y = df_test$output_predict_step_backward, prop.chisq = FALSE)
step_backward_test_lm_table_error = step_backward_test_lm_table$prop.tbl[1,2] + step_backward_test_lm_table$prop.tbl[2,1]
step_backward_test_lm_table_precision = step_backward_test_lm_table$prop.tbl[2,2]/(step_backward_test_lm_table$prop.tbl[2,2]+step_backward_test_lm_table$prop.tbl[1,2])
step_backward_test_lm_table_recall = step_backward_test_lm_table$prop.tbl[2,2]/(step_backward_test_lm_table$prop.tbl[2,2]+step_backward_test_lm_table$prop.tbl[2,1])

#Ridge Regression 
train_df = as.matrix(df[,c(1:85)])
test_df = as.matrix(df_test[,c(1:85)])

cv_ridge_model <- cv.glmnet(train_df,df$output_variable, alpha = 0, nfolds = 10)
plot(cv_ridge_model)
best_lamda_ridge = cv_ridge_model$lambda.min

coef(cv_ridge_model, s = "lambda.min")

predictions_test <- predict(cv_ridge_model, s = best_lamda_ridge, newx = test_df)
df_test$output_predict_ridge = predictions_test
summary(predictions_test)
df_test$output_predict_ridge = ifelse(df_test$output_predict_ridge<.5,0,1)
df_test$output_predict_ridge = as.factor(df_test$output_predict_ridge)
levels(df_test$output_predict_ridge)
ridge_test_lm_table = CrossTable(x=df_test$output_test, y = df_test$output_predict_ridge, prop.chisq = FALSE)
ridge_test_lm_table_error = ridge_test_lm_table$prop.tbl[1,2] + ridge_test_lm_table$prop.tbl[2,1]
ridge_test_lm_table_precision = ridge_test_lm_table$prop.tbl[2,2]/(ridge_test_lm_table$prop.tbl[2,2]+ridge_test_lm_table$prop.tbl[1,2])
ridge_test_lm_table_recall = ridge_test_lm_table$prop.tbl[2,2]/(ridge_test_lm_table$prop.tbl[2,2]+ridge_test_lm_table$prop.tbl[2,1])

#Lasso Regression

cv_lasso_model <- cv.glmnet(train_df,df$output_variable, alpha = 1, nfolds = 10)
plot(cv_lasso_model)
best_lamda_lasso = cv_lasso_model$lambda.min

coef(cv_lasso_model, s = "lambda.min")

predictions_test <- predict(cv_lasso_model, s = best_lamda_lasso, newx = test_df)
df_test$output_predict_lasso = predictions_test
summary(predictions_test)
df_test$output_predict_lasso = ifelse(df_test$output_predict_lasso<.5,0,1)
df_test$output_predict_lasso = as.factor(df_test$output_predict_lasso)
levels(df_test$output_predict_lasso)
lasso_test_lm_table = CrossTable(x=df_test$output_test, y = df_test$output_predict_lasso, prop.chisq = FALSE)
lasso_test_lm_table_error = lasso_test_lm_table$prop.tbl[1,2] + lasso_test_lm_table$prop.tbl[2,1]
lasso_test_lm_table_precision = lasso_test_lm_table$prop.tbl[2,2]/(lasso_test_lm_table$prop.tbl[2,2]+lasso_test_lm_table$prop.tbl[1,2])
lasso_test_lm_table_recall = lasso_test_lm_table$prop.tbl[2,2]/(lasso_test_lm_table$prop.tbl[2,2]+lasso_test_lm_table$prop.tbl[2,1])
