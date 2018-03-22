##########-----------------------------------------------------------------################
# NOTE: Models values can be different on your computer as random.seed() isn't set
# Also because the tune method randomly shuffles the data


#Read data files
temp = list.files(path ='Data/Variables_txt',  pattern="*.csv")
path = file.path('Data/Variables_txt/', temp)


Read = read.csv(path[1], header = F, nrows = 1178)[3]
View(Read)
for (i in 2:length(temp)){
  Read = cbind(Read, read.csv(path[i], header = F, fill = TRUE, nrows = 1178)[3])
} 

colnames(Read) <- temp



# Fill missing values with means
Read[Read==-99999] <- NA
colSums(is.na(Read))
column_means <- colMeans(Read, na.rm = T)
missing_value_indices <- which(is.na(Read), arr.ind=TRUE)
Read[missing_value_indices] <- column_means[missing_value_indices[,2]]

#Remove irrelavant feature
Read[Read$Closed_depression.csv <- NULL]
Read[Read$Slope.csv <- NULL]

#Remove inconsistent observation
Read[which(Read == 0, arr.ind = T)] <- NA
ind <- which(sapply(Read, is.na), arr.ind = T)
data <- Read[-ind[,1],]


View(data)

plot(Soil_moisture.csv~., data = data, col=2)


#Splitting the data into two sets the test being every second observation
len = dim(data)[1]

train <- data[seq(1, len, 2),]
test <- data[seq(2, len, 2),]


#Try Neural network model

library(neuralnet)

#Scaled features
maxs <- apply(train, 2, max) 
mins <- apply(train, 2, min)
train_scaled <- as.data.frame(scale(train, center = mins, scale = maxs - mins))

n <- names(train_scaled)
f <- as.formula(paste("Soil_moisture.csv ~", paste(n[!n %in% "Soil_moisture.csv"], collapse = " + ")))
nn <- neuralnet(f,data=train_scaled,hidden=c(3,2),linear.output=T)
f

#pdf("Neural_network.pdf")
plot(nn)
#dev.off()



library(hydroGOF) #contains mse, rmse metrics

maxs <- apply(test, 2, max) 
mins <- apply(test, 2, min)
test_scaled <- as.data.frame(scale(test, center = mins, scale = maxs - mins))

pr.nn <- compute(nn,test_scaled[,-10])
pr.nn <- pr.nn$net.result* (max(test$Soil_moisture.csv)-min(test$Soil_moisture.csv))+min(test$Soil_moisture.csv)

mse(sim = pr.nn, obs = data.frame(test$Soil_moisture.csv))



rmse(pr.nn, as.data.frame(test$Soil_moisture.csv))



#pdf("NN_prediction.pdf")
plot(test$Soil_moisture.csv,pr.nn,col='red',main='Neural Network',
     xlab='Actual', ylab = 'predicted', pch=18,cex=0.7)
#dev.off()
View(data)


#Cross validation for NN and save best model
cv.mse_error <- NULL
cv.rmse_error <- NULL
nn_models <- NULL
k <- 10
library(plyr) 
pbar <- create_progress_bar('text')
pbar$init(k)
for(i in 1:k){
  index <- sample(1:nrow(train),round(0.9*nrow(train)))
  train.cv <- train_scaled[index,]
  validate_scaled <- train_scaled[-index,]
  nn <- neuralnet(f,data=train.cv,hidden=c(5,2),linear.output=T)   
  nn_models[[i]] <- nn
  pr.nn <- compute(nn,validate_scaled[,-10])
  pr.nn <- pr.nn$net.result*(max(train$Soil_moisture.csv)-min(train$Soil_moisture.csv))+min(train$Soil_moisture.csv)   
  validate <- (validate_scaled$Soil_moisture.csv)*(max(train$Soil_moisture.csv)-min(train$Soil_moisture.csv))+min(train$Soil_moisture.csv)   
  cv.mse_error[i] <- mse(pr.nn, as.data.frame(validate))
  cv.rmse_error[i] <- rmse(pr.nn, as.data.frame(validate))
                     
  pbar$step()
}

boxplot(cv.mse_error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)

min(cv.mse_error)

#predict using the model with the minimum mse
nn_models
best_nn = nn_models[which.min(cv.mse_error)]
pr.nn_new <- compute(best_nn, test_scaled[,-10])
pr.nn_new <- pr.nn$net.result* (max(test$Soil_moisture.csv)-min(test$Soil_moisture.csv))+min(test$Soil_moisture.csv)

mse(sim = pr.nn_, obs = data.frame(test$Soil_moisture.csv))
rmse(pr.nn_, as.data.frame(test$Soil_moisture.csv))

plot(test$Soil_moisture.csv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)


#SVM Model

library(e1071)
svm_model <- svm(Soil_moisture.csv~., data = train_scaled, type='eps-regression')
pr.svm <- predict(svm_model, test_scaled[,-10])

pr.svm <- pr.svm*(max(test$Soil_moisture.csv)-min(test$Soil_moisture.csv))+min(test$Soil_moisture.csv)
test.bayes <- (test$Soil_moisture.csv)*(max(data$Soil_moisture.csv)-min(data$Soil_moisture.csv))+min(data$Soil_moisture.csv)

mse(pr.svm, test$Soil_moisture.csv)

plot(test$Soil_moisture.csv, pr.svm)

#pdf("SVM_prediction.pdf")
plot(test$Soil_moisture.csv,pr.svm,col='red',main='SVM',
     xlab='Actual', ylab = 'predicted', pch=18,cex=0.7)
#dev.off()


#Parameter tunning
svm_tune <- tune(svm, Soil_moisture.csv~., data = train_scaled, type='eps-regression',
                 ranges = list(epsilon = seq(0,1,0.01), cost = 2^(2:9))
)
print(svm_tune)

best_mod <- svm_tune$best.model
best_mod_pred <- predict(best_mod, test_scaled[,-10]) 

pr.svm_tune <- best_mod_pred*(max(test$Soil_moisture.csv)-min(test$Soil_moisture.csv))+min(test$Soil_moisture.csv)
rmse(pr.svm_tune, test$Soil_moisture.csv)


#pdf("SVM_tunning.pdf")
plot(svm_tune)
#dev.off()


  

#BART Model
library(BayesTree)
x.train <- subset(train_scaled, select = -Soil_moisture.csv)
y.train <- train_scaled$Soil_moisture.csv
x.test <- subset(test_scaled, select = -Soil_moisture.csv)


bayes <- bart(x.train = x.train, y.train = y.train, x.test = x.test)

summary(bayes)
pr.bayes <- bayes$yhat.test.mean*(max(test$Soil_moisture.csv)-min(test$Soil_moisture.csv))+min(test$Soil_moisture.csv)
test.bayes <- (test$Soil_moisture.csv)*(max(data$Soil_moisture.csv)-min(data$Soil_moisture.csv))+min(data$Soil_moisture.csv)

plot(bayes)
mse(pr.bayes, test$Soil_moisture.csv)
rmse(pr.bayes, test$Soil_moisture.csv)



#pdf("BART_prediction.pdf")
plot(test$Soil_moisture.csv,pr.bayes,col='red',main='BART',
     xlab='Actual', ylab = 'predicted', pch=18,cex=0.7)
#dev.off()

