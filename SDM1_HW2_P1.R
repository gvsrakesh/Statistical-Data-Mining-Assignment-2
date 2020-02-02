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

df = College
quartz()

df
df_final <- as.data.frame(lapply(df[,c(3:18)],function(x) ((x - mean(x))/(max(x)-min(x)))))
cols = c("Private","Apps")
df_final = cbind(df_final,df[,cols])
plot_correlation(df)

df_final = df_final[df_final$Apps <= 25000, ]
df_break = df_final[df_final$Apps <= 10000, ]
hist(df_break$Apps, breaks = 100, main = "Histogram of Applications for Universities less than 10k appliactions"
     ,xlab = "Applications below 10k")
colnames(df_final)
df_final = df_final[, !(colnames(df_final) %in% c("Enroll","F.Undergrad","Top10perc","Terminal")),
                    drop = FALSE]
kurtosis(df_final$Apps)
df_1h <- one_hot(as.data.table(df_final))

#defining function for calculating R squared as R2 is not working for some reason
rsq <- function (x, y) cor(x, y) ^ 2

set.seed(420)
training_samples_cont = df_1h$Apps %>% createDataPartition(p = 0.5, list = FALSE)
train_data_cont = df_1h[training_samples_cont, ]
test_data_cont = df_1h[-training_samples_cont, ]

names(df_final)

model = lm(Apps ~(.), data = train_data_cont)
summary(model)
#plot(model)

predictions_cont = model %>% predict(test_data_cont)
RMSE_test_LM = RMSE(predictions_cont, test_data_cont$Apps)
R2_test_LM = rsq(predictions_cont, test_data_cont$Apps)

#plot(model)

summary(train_data_cont)

train_data_cont_input = as.matrix(train_data_cont[,c(1:13)])
test_data_cont_input = as.matrix(test_data_cont[,c(1:13)])


#Ridge regression

cv_fit <- cv.glmnet(train_data_cont_input,train_data_cont$Apps, alpha = 0,nfolds = 10)
plot(cv_fit)
best_lamda_ridge = cv_fit$lambda.min
lambdas = cv_fit$lambda
min(lambdas)

lambdas <- seq(1000, 0, by = -10)
lamda_min_seq = min(lambdas)

cv_fit <- cv.glmnet(train_data_cont_input,train_data_cont$Apps, alpha = 0,nfolds = 5, lambda = lambdas)
plot(cv_fit)
best_lamda_ridge = cv_fit$lambda.min
lambdas = cv_fit$lambda

coef(cv_fit, s = "lambda.min")

y_predicted <- predict(cv_fit, s = best_lamda_ridge, newx = test_data_cont_input)
RMSE_test_ridge = RMSE(y_predicted, test_data_cont$Apps)
R2_test_ridge = as.numeric(rsq(y_predicted, test_data_cont$Apps))

#Lasso

cv_fit_lasso = cv.glmnet(train_data_cont_input,train_data_cont$Apps, alpha = 1, nfolds = 5)
plot(cv_fit_lasso)
best_lamda_lasso = cv_fit_lasso$lambda.min
std1_lamda_lasso = cv_fit_lasso$lambda.1se
#lambda_choose = (best_lamda_lasso+std1_lamda_lasso)/2
min(cv_fit_lasso$lambda)

plot(cv_fit_lasso)

y_predicted <- predict(cv_fit_lasso, s = best_lamda_lasso, newx = test_data_cont_input)
RMSE_test_lasso = RMSE(y_predicted, test_data_cont$Apps)
R2_test_lasso = as.numeric(rsq(y_predicted, test_data_cont$Apps))
coef(cv_fit_lasso, s = "lambda.min")

y_predicted <- predict(cv_fit_lasso, s = std1_lamda_lasso, newx = test_data_cont_input)
RMSE_test_lasso_se1 = RMSE(y_predicted, test_data_cont$Apps)
R2_test_lasso_se1 = as.numeric(rsq(y_predicted, test_data_cont$Apps))
coef(cv_fit_lasso, s = "lambda.1se")

#PCR
quartz()
cv_fit_pcr <- pcr(train_data_cont$Apps~., data = train_data_cont, scale = TRUE, validation = "CV")
validationplot(cv_fit_pcr, val.type = "R2")
validationplot(cv_fit_pcr, val.type="RMSEP")

y_predicted <- predict(cv_fit_pcr, test_data_cont, ncomp = 2)
RMSE_test_pcr_2 = RMSE(y_predicted, test_data_cont$Apps)
R2_test_pcr_2 = rsq(y_predicted, test_data_cont$Apps)

y_predicted <- predict(cv_fit_pcr, test_data_cont, ncomp = 10)
RMSE_test_pcr_10 = RMSE(y_predicted, test_data_cont$Apps)
R2_test_pcr_10 = rsq(y_predicted, test_data_cont$Apps)

#PLS
quartz()
cv_fit_pls = plsr(train_data_cont$Apps~., data=train_data_cont, validation="CV")
validationplot(cv_fit_pls, val.type = "R2")
validationplot(cv_fit_pls, val.type = "RMSEP")

y_predicted <- predict(cv_fit_pls, test_data_cont, ncomp = 4)
RMSE_test_pls_4 = RMSE(y_predicted, test_data_cont$Apps)
R2_test_pls_4 = rsq(y_predicted, test_data_cont$Apps)

y_predicted <- predict(cv_fit_pls, test_data_cont, ncomp = 6)
RMSE_test_pls_6 = RMSE(y_predicted, test_data_cont$Apps)
R2_test_pls_6 = rsq(y_predicted, test_data_cont$Apps)

