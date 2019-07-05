# Load Data
Adv_sales <- read.csv(file.choose())


###########Loading Library################
library(ggplot2) # visualization
library(ggthemes) # visualization
library(scales) # visualization
library(dplyr) # data manipulation
library(mice) # imputation
library(class) # imputation with kNN
library(glmnet) #For ridge and lasso
library(caret) # Model
library(gpairs) # visualization
library(corrplot) # corelation
###########Loading Library################





###########Data exploration################
str(Adv_sales)
summary(Adv_sales)

plot(Adv_sales)
gpairs(Adv_sales)


#Check relation of each attribute with th target.
#Sales and store
ggplot(Adv_sales, aes(x = sales, y = store, colour = store)) +
  geom_point() + 
  ggtitle("Relation between sales and advertisement spending in store") +
  geom_smooth()

#Sales and billboard
ggplot(Adv_sales, aes(x = sales, y = billboard, colour = billboard)) +
  geom_point() + 
  ggtitle("Relation between sales and advertisement spending on billboard") +
  geom_smooth()

#Sales and printout
ggplot(Adv_sales, aes(x = sales, y = printout, colour = printout)) +
  geom_point() + 
  ggtitle("Relation between sales and advertisement spending on printout") +
  geom_smooth()


#Sales and sat
ggplot(Adv_sales, aes(x = sales, y = sat, colour = sat)) +
  geom_point() + 
  ggtitle("Relation between sales and satisfaction level") +
  geom_smooth()


#Sales and comp
ggplot(Adv_sales, aes(x = sales, y = comp, colour = comp)) +
  geom_point() + 
  ggtitle("Relation between sales and competitors advertisement spending") +
  geom_smooth()


#Sales and price
ggplot(Adv_sales, aes(x = sales, y = price, colour = price)) +
  ggtitle("Relation between sales and price") +
  geom_point() + 
  geom_smooth()


#Corelation of the dataset
par(mfrow=c(1,1))
corrplot.mixed(cor(Adv_sales[ , c(2, 3:8)]), upper="ellipse")
#Additinal data exploration
###########Data exploration################


###########Split Train and Test Data#######
set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime
inTrain <- createDataPartition(y = Adv_sales$sales,
                               p = 750/1000, list = FALSE)
training <- Adv_sales[ inTrain,]
testing <- Adv_sales[ -inTrain,]
###########Split Train and Test Data#######



##########Model Building, prediction and validation########### # LM MODEL ONLY
#Model with one feature "price"
m1.train<-lm(sales~price, data=training)
summary(m1.train)$r.squared # 0.05791981
m1.test<-predict(m1.train,testing)
SSE1 = sum((testing$sales - m1.test)^2) # Explained variation
Rsq1=1 - SSE1/SST
Rsq1 # 0.06051557
mean(abs(m1.test-testing$sales)/testing$sales*100) # 18.43887

#Model with two feature "price + store"
m2.train<-lm(sales~price+store, data=training)
summary(m2.train)$r.squared #0.3257447
m2.test<-predict(m2.train,testing)
SSE2 = sum((testing$sales - m2.test)^2) # Explained variation
Rsq2 =1 - SSE2/SST
Rsq2 # 0.3188413
mean(abs(m2.test-testing$sales)/testing$sales*100) # 15.67223

#Model with three feature "price + store + billboard"
m3.train<-lm(sales~price+store+ billboard, data=training)
summary(m3.train)$r.squared #0.8378117
m3.test<-predict(m3.train,testing)
SSE3 = sum((testing$sales - m3.test)^2) # Explained variation
Rsq3 =1 - SSE3/SST
Rsq3 # 0.8597913
mean(abs(m3.test-testing$sales)/testing$sales*100) # 7.025053

#Model with four feature "price + store + billboard + printout"
m4.train<-lm(sales~price+store+ billboard+printout, data=training)
summary(m4.train)$r.squared # 0.8378213
m4.test<-predict(m4.train,testing)
SSE4 = sum((testing$sales - m4.test)^2) # Explained variation
Rsq4 =1 - SSE4/SST
Rsq4 # 0.8597317
mean(abs(m4.test-testing$sales)/testing$sales*100) # 7.026433

#Model with five feature "price + store + billboard + printout + sat"
m5.train<-lm(sales~price+store+ billboard+printout+sat, data=training)
summary(m5.train)$r.squared # 0.9124497
m5.test<-predict(m5.train,testing)
SSE5 = sum((testing$sales - m5.test)^2) # Explained variation
Rsq5 =1 - SSE5/SST
Rsq5 # 0.9188513
mean(abs(m5.test-testing$sales)/testing$sales*100) # 5.358562

#Model with six feature "price + store + billboard + printout + sat + comp"
m6.train<-lm(sales~price+store+ billboard+printout+sat+comp, data=training)
summary(m6.train)$r.squared # 0.9204201
m6.test<-predict(m6.train,testing)
SSE6 = sum((testing$sales - m6.test)^2) # Explained variation
Rsq6 =1 - SSE6/SST
Rsq6 # 0.9192729
mean(abs(m6.test-testing$sales)/testing$sales*100) # 5.375458

###### Interaction effects: : denotes an interaction term between regressors.
#Model with interactions: price
m7.train<-lm(sales~price+store+ billboard+printout+sat+comp + store:price + billboard:price + printout:price+ sat:price + comp:price , data=training)
summary(m7.train)$r.squared # 0.9208608
m7.test<-predict(m7.train,testing)
SSE7 = sum((testing$sales - m7.test)^2) # Explained variation
Rsq7 =1 - SSE7/SST
Rsq7 # 0.9188576
mean(abs(m7.test-testing$sales)/testing$sales*100) # 5.394289

#Model with interactions: store
m8.train<-lm(sales~price+store+ billboard+printout+sat+comp + price:store + billboard:store + printout:store+ sat:store + comp:store , data=training)
summary(m8.train)$r.squared # 0.9244214
m8.test<-predict(m8.train,testing)
SSE8 = sum((testing$sales - m8.test)^2) # Explained variation
Rsq8 =1 - SSE8/SST
Rsq8 # 0.9301727
mean(abs(m8.test-testing$sales)/testing$sales*100) # 5.08351

#Model with interactions: billboard
m9.train<-lm(sales~price+store+ billboard+printout+sat+comp + price:billboard + store:billboard + printout:billboard+ sat:billboard + comp:billboard , data=training)
summary(m9.train)$r.squared # 0.9250358
m9.test<-predict(m9.train,testing)
SSE9 = sum((testing$sales - m9.test)^2) # Explained variation
Rsq9 =1 - SSE9/SST
Rsq9 # 0.9292253
mean(abs(m9.test-testing$sales)/testing$sales*100) # 5.097023

#Model with interactions: printout
m10.train<-lm(sales~price+store+ billboard+printout+sat+comp + price:printout + store:printout + billboard:printout + sat:printout + comp:printout , data=training)
summary(m10.train)$r.squared # 0.9211429
m10.test<-predict(m10.train,testing)
SSE10 = sum((testing$sales - m10.test)^2) # Explained variation
Rsq10 =1 - SSE10/SST
Rsq10 # 0.9187207
mean(abs(m10.test-testing$sales)/testing$sales*100) # 5.389549

#Model with interactions: Satisfaction
m11.train<-lm(sales~price+store+ billboard+printout+sat+comp + price:sat + store:sat + billboard:sat + printout:sat + comp:sat , data=training)
summary(m11.train)$r.squared # 0.9207655
m11.test<-predict(m11.train,testing)
SSE11 = sum((testing$sales - m11.test)^2) # Explained variation
Rsq11 =1 - SSE11/SST
Rsq11 # 0.9186277
mean(abs(m11.test-testing$sales)/testing$sales*100) # 5.388559

#Model with interactions: competitors advertisement spending
m12.train<-lm(sales~price+store+ billboard+printout+sat+comp + price:comp + store:comp + billboard:comp + printout:comp + sat:comp , data=training)
summary(m12.train)$r.squared # 0.9205291
m12.test<-predict(m12.train,testing)
SSE12 = sum((testing$sales - m12.test)^2) # Explained variation
Rsq12 =1 - SSE12/SST
Rsq12 # 0.9192619
mean(abs(m12.test-testing$sales)/testing$sales*100) # 5.386198


#Model with interactions: X:Z
m13.train<-lm(sales~price+store+ billboard+printout+sat+comp + price:store + billboard:store + printout:store+ sat:store + comp:store 
              +price:billboard + store:billboard + printout:billboard+ sat:billboard + comp:billboard, data=training)
summary(m13.train)$r.squared # 0.9250916
m13.test<-predict(m13.train,testing)
SSE13 = sum((testing$sales - m13.test)^2) # Explained variation
Rsq13 =1 - SSE13/SST
Rsq13 # 0.9291586
mean(abs(m13.test-testing$sales)/testing$sales*100) # 5.108118

#Model with interactions: X*Z
m14.train<-lm(sales~price+store* billboard+printout+sat+comp, data=training)
summary(m14.train)$r.squared # 0.924383
m14.test<-predict(m14.train,testing)
SSE14 = sum((testing$sales - m14.test)^2) # Explained variation
Rsq14 =1 - SSE14/SST
Rsq14 # 0.9302663
mean(abs(m14.test-testing$sales)/testing$sales*100) # 5.071738

#Reduced model - drop insignificant terms Printout
m15.train<-lm(sales~price+store* billboard+sat+comp, data=training)
summary(m15.train)$r.squared # 0.9243804
m15.test<-predict(m15.train,testing)
SSE15 = sum((testing$sales - m15.test)^2) # Explained variation
Rsq15 =1 - SSE15/SST
Rsq15 #0.9302154
mean(abs(m15.test-testing$sales)/testing$sales*100) # 5.073923 # This is the best model as R^ and mape is the best 
##########Model Building, prediction and validation########### # LM MODEL ONLY


###################LASSO MODEL#############################
T_training<- Adv_sales[1:750,]
T_testing<- Adv_sales[751:1000,]
y<-log(T_training$sales)
X<-model.matrix(ID~price+store+ billboard+printout+sat+comp, Adv_sales)[,-1]
X<-cbind(Adv_sales$ID,X)

# split X into testing, trainig/holdout and prediction as before
X.training<-subset(X,X[,1]<=750)
X.testing<-subset(X, (X[,1]>=751 & X[,1]<=1000))


#LASSO (alpha=1)
lasso.fit<-glmnet(x = X.training, y = y, alpha = 1)
plot(lasso.fit, xvar = "lambda")

#selecting the best penalty lambda
crossval <-  cv.glmnet(x = X.training, y = y, alpha = 1) #create cross-validation data
plot(crossval)
penalty.lasso <- crossval$lambda.min #determine optimal penalty parameter, lambda
log(penalty.lasso) #see where it was on the graph
plot(crossval,xlim=c(-8.5,-6),ylim=c(0.006,0.008)) # lets zoom-in
lasso.opt.fit <-glmnet(x = X.training, y = y, alpha = 1, lambda = penalty.lasso) #estimate the model with the optimal penalty
coef(lasso.opt.fit) #resultant model coefficients

# predicting the performance on the testing set
lasso.testing <- exp(predict(lasso.opt.fit, s = penalty.lasso, newx =X.testing))
mean(abs(lasso.testing-T_testing$sales)/T_testing$sales*100) #calculate and display MAPE 5.116034

#ridge (alpha=0)
ridge.fit<-glmnet(x = X.training, y = y, alpha = 0)
plot(ridge.fit, xvar = "lambda")

#selecting the best penalty lambda
crossval <-  cv.glmnet(x = X.training, y = y, alpha = 0)
plot(crossval)
penalty.ridge <- crossval$lambda.min 
log(penalty.ridge) 
ridge.opt.fit <-glmnet(x = X.training, y = y, alpha = 0, lambda = penalty.ridge) #estimate the model with that
coef(ridge.opt.fit)

ridge.testing <- exp(predict(ridge.opt.fit, s = penalty.ridge, newx =X.testing))
mean(abs(ridge.testing-T_testing$sales)/T_testing$sales*100) # 5.226102
###################LASSO MODEL#############################

#Adding New features
Adv_sales <- mutate(Adv_sales, store_billboard = store + billboard)
Adv_sales <- mutate(Adv_sales, store_printout = store + printout)
Adv_sales <- mutate(Adv_sales, billboard_printout = billboard + printout)
Adv_sales <- mutate(Adv_sales, int_store_T_billboard = billboard * store)

##############Data Split#############
# Divide data set in train 75% and test 25%
set.seed(77850) #set a random number generation seed to ensure that the split is the same everytime
library(caret)
inTrain <- createDataPartition(y = Adv_sales$sales,
                               p = 750/1000, list = FALSE)
training <- Adv_sales[ inTrain,]
testing <- Adv_sales[ -inTrain,]
##############Data Split#############


###################ALL 13 methods of Random Forest########################
RM1 <- trainControl(method = "boot", number = 100, search = "random")
RM2 <- trainControl(method = "boot632", number = 100, search = "random")
RM3 <- trainControl(method = "optimise_boot", number = 100, search = "random")
RM4 <- trainControl(method = "boot_all", number = 100, search = "random")
RM5 <- trainControl(method = "cv", number = 100, search = "random")
RM6 <- trainControl(method = "repeatedcv", number = 100, search = "random")
RM7 <- trainControl(method = "LOOCV", number = 100, search = "random")
RM8 <- trainControl(method = "LGOCV", number = 100, search = "random")
RM9 <- trainControl(method = "none", number = 100, search = "random")
RM10 <- trainControl(method = "oob", number = 100, search = "random")
RM11 <- trainControl(method = "adaptive_cv", number = 100, search = "random")
RM12 <- trainControl(method = "adaptive_boot", number = 100, search = "random")
RM13 <- trainControl(method = "adaptive_LGOCV", number = 100, search = "random")




set.seed(77850)
Mod_boot = train(sales~ ., data = training, method = "rf", trCotrol = "RM1", importance = TRUE, proximity = TRUE)
Mod_boot632 = train(sales~ ., data = training, method = "rf", trCotrol = "RM2", importance = TRUE, proximity = TRUE)
Mod_optimise_boot = train(sales~ ., data = training, method = "rf", trCotrol = "RM3", importance = TRUE, proximity = TRUE)
Mod_boot_all = train(sales~ ., data = training, method = "rf", trCotrol = "RM4", importance = TRUE, proximity = TRUE)
Mod_cv = train(sales~ ., data = training, method = "rf", trCotrol = "RM5", importance = TRUE, proximity = TRUE)
Mod_repeatedcv = train(sales~ ., data = training, method = "rf", trCotrol = "RM6", importance = TRUE, proximity = TRUE)
Mod_LOOCV = train(sales~ ., data = training, method = "rf", trCotrol = "RM7", importance = TRUE, proximity = TRUE)
Mod_LGOCV = train(sales~ ., data = training, method = "rf", trCotrol = "RM8", importance = TRUE, proximity = TRUE)
Mod_none = train(sales~ ., data = training, method = "rf", trCotrol = "RM9", importance = TRUE, proximity = TRUE)
Mod_oob = train(sales~ ., data = training, method = "rf", trCotrol = "RM10", importance = TRUE, proximity = TRUE)
Mod_adaptive_cv = train(sales~ ., data = training, method = "rf", trCotrol = "RM11", importance = TRUE, proximity = TRUE)
Mod_adaptive_boot = train(sales~ ., data = training, method = "rf", trCotrol = "RM12", importance = TRUE, proximity = TRUE)
Mod_adaptive_LGOCV = train(sales~ ., data = training, method = "rf", trCotrol = "RM13", importance = TRUE, proximity = TRUE)




results <- resamples(list(boot = Mod_boot, boot632 = Mod_boot632, optimise_boot = Mod_optimise_boot,
                          boot_all = Mod_boot_all, cv = Mod_cv, repeatedcv = Mod_repeatedcv,
                          LOOCV = Mod_LOOCV, LGOCV = Mod_LGOCV, none = Mod_none, oob = Mod_oob,
                          adaptive_cv = Mod_adaptive_cv, adaptive_boot = Mod_adaptive_boot, adaptive_LGOCV = Mod_adaptive_LGOCV
))

summary(results)
dotplot(results)

#Best sampling method
random_forest_model <- train(sales~ ., data = training, method = "rf", trCotrol = "RM11", importance = TRUE, proximity = TRUE)

#Predict and avaluation on testing
Predict_on_testing = predict(random_forest_model, newdata =subset(testing, select = -c(testing$sales)))


#Mean Squared Test Error
RMSE <- sqrt(sum((Predict_on_testing - testing$sales)^2)/length(Predict_on_testing))
print(RMSE)
print(RMSE/mean(testing$sales)) # 0.07575169

#MAPE
MAPE = mean(abs(Predict_on_testing-testing$sales)/testing$sales)
print(MAPE) # 0.0639475
##############################################################Random forest#############################################




