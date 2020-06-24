library(data.table)
library(ggplot2)
library(caTools)
library(readr)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)
library(DiagrammeR)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(dplyr)
library(randomForest)
library(forecastHybrid)

current_working_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_working_dir)


# ==================== Data Cleaning ====================


# Reducing data size by removing redundant & meaningless rows
datatable <- fread("Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv", stringsAsFactors = TRUE)
relevant.cols <- c("IncidntNum", "Category", "Descript", "DayOfWeek", "Date", "Time", "PdDistrict", "Resolution", "Address", "X", "Y", "Location", "PdId")
relevant.dt <- datatable[, relevant.cols, with = FALSE]

# IncidntNum & PdId are unique identifiers
relevant.dt$IncidntNum <- NULL
relevant.dt$PdId <- NULL

# Too many distinct addresses
relevant.dt$Address <- NULL

summary(relevant.dt)
fwrite(relevant.dt, "SFCrimeData.csv")


# ============================================================
# Section 2.1 - Predictive Patrolling, Smarter Patrolling
# ============================================================


data <- fread("SFCrimeData.csv", stringsAsFactors = TRUE)


# ==================== Data Processing ====================


# Filter for blue collar crimes, which are more easily observable
blue_collar_crimes = c("VANDALISM", "LARCENY/THEFT", "STOLEN PROPERTY", "ROBBERY", 
                     "DRIVING UNDER THE INFLUENCE", "DISORDERLY CONDUCT", 
                     "LIQUOR LAWS", "VEHICLE THEFT", "ASSAULT", "KIDNAPPING", 
                     "TRESPASS", "ARSON", "RECOVERED VEHICLE")

data <- data[Category %in% blue_collar_crimes]
data <- droplevels(data)

# Check for incomplete records
data[!complete.cases(data)]

# Extract Month & Day from Date
data$Date <- as.Date(data$Date, "%m/%d/%Y")
data$Month <- month(data$Date)
data$Day <- mday(data$Date)
data$Date <- NULL

# Extract Hour from Time
data$Hour <- substr(data$Time, 1, 2)
data$Hour <- as.integer(data$Hour)
data$Time <- NULL

# Remove irrelevant columns for our prediction
data$Location <- NULL       # Location is just X & Y combined
data$Descript <- NULL       # This describes Category, our predicted variable
data$Resolution <- NULL     # Resolution is an after-event outcome.

# Slice San Francisco into a 10 x 10 grid map
n <- 10

summary(data$X)
summary(data$Y)
data <- data[Y != 90,]

longitude_grids <- c()
for (i in 1:(n + 1)){
  longitude_grids[i] <- min(data$X) + (i - 1) * ((max(data$X) - min(data$X))/n)
}

latitude_grids <- c()
for (i in 1:(n + 1)){
  latitude_grids[i] <- min(data$Y) + (i - 1) * ((max(data$Y) - min(data$Y))/n)
}

region <- data.table()
region$regionID <- 1:n^2
region$bottomLeftLongitude <- rep(longitude_grids[1:n], times = n)
region$bottomLeftLatitude <- rep(latitude_grids[1:n], times = rep(n, n))
region$topRightLongitude <- rep(longitude_grids[2:(n+1)], times = n)
region$topRightLatitude <- rep(latitude_grids[2:(n+1)], times = rep(n, n))

for (regionid in c(1:dim(region)[1])){
  data[X >= region$bottomLeftLongitude[regionid] & X < region$topRightLongitude[regionid]
       & Y >= region$bottomLeftLatitude[regionid] & Y < region$topRightLatitude[regionid], 
       regionId := regionid]
}

# Make sure every row has a regId
data[is.na(data$regionId) == TRUE]
data[is.na(data$regionId), regionId := n^2]

# Now we can remove X & Y
data$X <- NULL
data$Y <- NULL

# Cleaning up redundant data
rm(blue_collar_crimes, i, latitude_grids, longitude_grids, relevant.cols,
   n, regionid)


# ==================== Model Building (XGBoost) ====================


set.seed(2020)
train <- sample.split(data$Category, SplitRatio = 0.7)
trainset <- subset(data, train == TRUE)
testset <- subset(data, train == FALSE)
rm(train)

# Double check for NAs
table(is.na(trainset))
table(is.na(testset))

train_labels <- as.integer(trainset$Category) - 1
test_labels <- as.integer(testset$Category) - 1

new_trainset <- model.matrix(~.+0, data = trainset[, -c("Category"), with=F]) 
new_testset <- model.matrix(~.+0, data = testset[, -c("Category"), with=F])

# For XGBoost, we use xgb.DMatrix to convert data table into a matrix
dtrain <- xgb.DMatrix(data = new_trainset, label = train_labels) 
dtest <- xgb.DMatrix(data = new_testset, label= test_labels)

# Cleaning up redundant data
rm(trainset, new_trainset, new_testset)

watchlist <- list(train = dtrain, test = dtest)

# Parameters
# Note that num_class = 13, since there are 13 categories of white collar crime
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = length(unique(data$Category)))

set.seed(2020)
# eXtreme Gradient Boosting Model
# We can go up to 1000 iterations
# But if there has been no improvement for 10 rounds, we will stop early. 
xgb_model <- xgb.train(params = xgb_params,
                       data = dtrain,
                       nrounds = 1000,
                       watchlist = watchlist, 
                       early_stopping_rounds = 10)

e <- data.frame(xgb_model$evaluation_log)
colnames(e) <- c("Iteration Number", "TrainSet mlogloss", "TestSet mlogloss")
plot(e$`Iteration Number`, e$`TrainSet mlogloss`, type = 'l', col = 'blue', xlab = "Iteration Number", ylab = "mLogLoss")
lines(e$`Iteration Number`, e$`TestSet mlogloss`, col = 'red', xlab = "Iteration Number", ylab = "mLogLoss")
legend("topright", legend = c("TrainSet", "TestSet"), col = c("blue", "red"), lty = c(1,1), cex = 0.8)
rm(e, watchlist, dtrain)


# ==================== Model Testing & Interpretation (XGBoost) ====================


pred.xgboost <- predict(xgb_model, newdata = dtest, reshape = TRUE)
pred.xgboost <- as.data.table(pred.xgboost)
colnames(pred.xgboost) = levels(data$Category)

top.3.xgboost <- data.table()
top.3.xgboost$first = apply(pred.xgboost, 1, function(x) colnames(pred.xgboost)[which.max(x)])
top.3.xgboost$first_prob = apply(pred.xgboost, 1, function(x) max(x))
top.3.xgboost$second <- apply(pred.xgboost, 1, function(x) colnames(pred.xgboost)[which(x == sort(x, decreasing = TRUE)[2])])
top.3.xgboost$second_porb <- apply(pred.xgboost, 1, function(x) sort(x, decreasing = TRUE)[2])
top.3.xgboost$third <- apply(pred.xgboost, 1, function(x) colnames(pred.xgboost)[which(x == sort(x, decreasing = TRUE)[3])])
top.3.xgboost$third_porb <- apply(pred.xgboost, 1, function(x) sort(x, decreasing = TRUE)[3])

top.3.xgboost$actual <- testset$Category

top.3.xgboost$result = 0
top.3.xgboost[first == actual, result := 1]
top.3.xgboost[second == actual, result := 1]
top.3.xgboost[third == actual, result := 1]

mean(top.3.xgboost$result)

importance_matrix <- xgb.importance(model = xgb_model)
importance_matrix <- data.table(importance_matrix)
importance_matrix <- head(importance_matrix, 5)
xgb.plot.importance(importance_matrix = importance_matrix)

# How an individual tree in XGBoost looks like:
xgb.plot.tree(model = xgb_model, trees = 0, show_node_id = TRUE)

# The number of trees = number of iterations * number of classes (in Category)
total_trees <- xgb_model$niter * xgb_model$params$num_class

xgb.plot.tree(model = xgb_model, trees = total_trees - 1, show_node_id = TRUE)

rm(train_labels, test_labels, dtest)


# ============================================================
# Section 2.2 - Predictive Resolution for Crime Allocation
# ============================================================


dt <- fread("SFCrimeData.csv", stringsAsFactors = TRUE)
dt$Location <- NULL

par(mar=c(4,23,4,10))
barplot(sort(summary(dt$Resolution)), xlim = c(0, 1400000), main = "Resolution Counts", xlab = "Count", col = "darkred", horiz = TRUE, las = 1)
par(mar = c(4, 4, 4, 4))


# ==================== Data Processing ====================


dt$Date <- as.Date(dt$Date, "%m/%d/%Y")
dt$Year <- year(dt$Date)
dt$Month <- month(dt$Date)
dt$Day <- mday(dt$Date)
dt$Date <- NULL
dt$Category <- factor(dt$Category)
dt$PdDistrict <- factor(dt$PdDistrict)
dt$DayOfWeek <- factor(dt$DayOfWeek)
dt$Descript <- factor(dt$Descript)

dt[Resolution != "NONE", Solvable := "YES"]
dt[Resolution == "NONE", Solvable := "NO"]
dt$Solvable <- factor(dt$Solvable)

dt$Hour <- substr(dt$Time,0,2)
dt$Hour <- as.numeric(dt$Hour)

dt[,Time:=NULL]
dt[,Resolution:=NULL]
sapply(dt, class)

coul <- brewer.pal(5, "Set2") 
barplot(sort(summary(dt$Solvable)), xlim= c(0, 1400000), main = "Solvable Counts", xlab= "Count", col= coul, horiz = TRUE, las =1)


# ==================== CART ====================


# default cp = 0.01. Set cp = 0 to guarantee no pruning.
set.seed(2020)
cart <- rpart(Solvable ~ ., data = dt, method = 'class', control = rpart.control(minsplit = 5000, cp = 0))
printcp(cart)
plotcp(cart)

# Compute min CVerror + 1SE in maximal tree cart1.
CVerror.cap <- cart$cptable[which.min(cart$cptable[,"xerror"]), "xerror"] + cart$cptable[which.min(cart$cptable[,"xerror"]), "xstd"]

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree cart.
i <- 1; j<- 4
while (cart$cptable[i,j] > CVerror.cap) { 
  i <- i + 1 
}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp.opt = ifelse(i > 1, sqrt(cart$cptable[i,1] * cart$cptable[i-1,1]), 1)

cart.new <- prune(cart, cp = cp.opt)
rpart.plot(cart.new)
cart.new$variable.importance

predict.train<- predict(cart.new, type = 'class')
table.train <- table(dt$Solvable, predict.train)
round(prop.table(table.train),3)
cat("Overall Accuracy Rate: ", round(mean(predict.train == dt$Solvable),3), "\n")

# Extension - Application purposes for the Police
testset <- sample_n(dt, 200) #just for an example dataset
testset$Solvable <- NULL
predict.test<- predict(cart.new, newdata = testset,type = 'prob')
n_classes <- 1
PredictedYES <- predict.test[,2]
predictedClass <- apply(predict.test,1,function(xx)head(names(sort(xx, decreasing=T)), n_classes))
testset$PredictedYES <- PredictedYES
testset$PredictedClass <- predictedClass
head(testset,25)


# ==================== Logistic Regression ====================


# Train-Test Split (logit) 
sampled.dt <- sample_n(dt, 80000) # due to memory requirements

train <- sample.split(Y = sampled.dt$Solvable, SplitRatio = 0.75)
trainset <- subset(sampled.dt, train == T)
testset <- subset(sampled.dt, train == F)

summary(trainset)
prop.table(table(trainset$Solvable))
summary(testset)
prop.table(table(testset$Solvable))

# Model 1
set.seed(2020)
logit <- glm(Solvable ~ . , family = binomial(), data = trainset)
options(max.print=1000000)
summary(logit)

OR <- exp(coef(logit))
OR

OR.CI <- exp(confint(logit))
OR.CI

threshold <- 0.5

prob.train <- predict(logit, type = 'response')
predict.train <- ifelse(prob.train > threshold, "YES", "NO")
table.train <- table(trainset$Solvable, predict.train)
round(prop.table(table.train),3)
cat("Trainset Overall Accuracy Rate: ", round(mean(predict.train == trainset$Solvable),3), "\n")

logit$xlevels[["Category"]] <- union(logit$xlevels[["Category"]], levels(testset$Category))
logit$xlevels[["Descript"]] <- union(logit$xlevels[["Descript"]], levels(testset$Descript))

prob.test <- predict(logit, newdata = testset)
predict.test <- ifelse(prob.test > threshold, "YES", "NO")
table.test <- table(testset$Solvable, predict.test)
round(prop.table(table.test),3)
cat("Testset Overall Accuracy Rate: ", round(mean(predict.test == testset$Solvable),3), "\n")
# Significant variables are in order: Description, Category, Pd District, Year(but should be removed because older cases are more likely to already be solved)

# Train-Test Split (logit2)
sampled.dt <- sample_n(dt, 80000) # due to memory requirements
sampled.dt$Year<-NULL
sampled.dt$X<-NULL
sampled.dt$Y<-NULL
sampled.dt$DayOfWeek<-NULL
sampled.dt$Hour<-NULL
sampled.dt$Month<-NULL
sampled.dt$Day<-NULL
sampled.dt$Time<-NULL
train <- sample.split(Y = sampled.dt$Solvable, SplitRatio = 0.75)
trainset <- subset(sampled.dt, train == T)
testset <- subset(sampled.dt, train == F)

summary(trainset)
prop.table(table(trainset$Solvable))
summary(testset)
prop.table(table(testset$Solvable))

# Model 2
set.seed(2020)
logit <- glm(Solvable ~ . , family = binomial(), data = trainset)
options(max.print=1000000)
summary(logit)

OR <- exp(coef(logit))
OR

OR.CI <- exp(confint(logit))
OR.CI

threshold <- 0.5

prob.train <- predict(logit, type = 'response')
predict.train <- ifelse(prob.train > threshold, "YES", "NO")
table.train <- table(trainset$Solvable, predict.train)
round(prop.table(table.train),3)
cat("Trainset Overall Accuracy Rate: ", round(mean(predict.train == trainset$Solvable),3), "\n")

logit$xlevels[["Category"]] <- union(logit$xlevels[["Category"]], levels(testset$Category))
logit$xlevels[["Descript"]] <- union(logit$xlevels[["Descript"]], levels(testset$Descript))

prob.test <- predict(logit, newdata = testset)
predict.test <- ifelse(prob.test > threshold, "YES", "NO")
table.test <- table(testset$Solvable, predict.test)
round(prop.table(table.test),3)
cat("Testset Overall Accuracy Rate: ", round(mean(predict.test == testset$Solvable),3), "\n")


# ==================== XGBoost ====================


# Train-Test Split (XGBoost) & Processing
solvable = dt$Solvable
label = as.integer(dt$Solvable) - 1
dt$Category <- as.integer(dt$Category) - 1
dt$Descript <- as.integer(dt$Descript) - 1
dt$DayOfWeek <- as.integer(dt$DayOfWeek) - 1
dt$PdDistrict <- as.integer(dt$PdDistrict) - 1
dt$Time <- parse_time(dt$Time, "%H:%M")
dt$Time <- as.integer(dt$Time) - 1
dt$Solvable <- NULL

n = nrow(dt)
train.index = sample(n,floor(0.75*n))
train.data = as.matrix(dt[train.index,])
train.label = label[train.index]
test.data = as.matrix(dt[-train.index,])
test.label = label[-train.index]

xgb.train = xgb.DMatrix(data=train.data,label=train.label)
xgb.test = xgb.DMatrix(data=test.data,label=test.label)


num_class = length(levels(solvable)) # Yes or No

# Model
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = num_class) 

watchlist <- list(train = xgb.train, test = xgb.test)

set.seed(2020)
# eXtreme Gradient Boosting Model
bst_model <- xgb.train(params = xgb_params,
                       data = xgb.train,
                       nrounds = 500,      # 500 iterations
                       watchlist = watchlist)

bst_model

# Trainset Results
xgb.pred.train = predict(bst_model, train.data, reshape=T)
xgb.pred.train = as.data.frame(xgb.pred.train)
colnames(xgb.pred.train) = levels(solvable)

xgb.pred.train$prediction = apply(xgb.pred.train,1,function(x) colnames(xgb.pred.train)[which.max(x)])
xgb.pred.train$label = levels(solvable)[train.label+1]

table.train <- table(xgb.pred.train$prediction, xgb.pred.train$label)
round(prop.table(table.train),3)
result = sum(xgb.pred.train$prediction==xgb.pred.train$label)/nrow(xgb.pred.train)
print(paste("Train Final Accuracy =",sprintf("%1.2f%%", 100*result)))

# Testset Results
xgb.pred = predict(bst_model,test.data,reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = levels(solvable)

xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label = levels(solvable)[test.label+1]
xgb.pred$NO <- NULL

table.test <- table(xgb.pred$prediction, xgb.pred$label)
round(prop.table(table.test),3)
result = sum(xgb.pred$prediction==xgb.pred$label)/nrow(xgb.pred)
print(paste("Test Final Accuracy =",sprintf("%1.2f%%", 100*result)))

# Extension - Application purposes for the Police
xgb.pred


# ============================================================
# Section 3 - Forecasting the Number of Future Crimes
# ============================================================


d <- fread("SFCrimeData.csv", stringsAsFactors = TRUE)


# ==================== Data Preparation ====================


### --- Cases involving Juveniles/Children ---

# Subsetting the cases with "JUVENILE" in the description
sum(grepl("JUVENILE", d$Descript) == "TRUE") # 23990 cases
JVNL <- d[grep("JUVENILE", d$Descript)]
summary(JVNL$Descript) # Mix of Juvenile VICTIMS & PERPETRATORS
# Need to note / Anomalies:
#   SELLING RESTRICTED GLUE TO JUVENILES (5 cases) - to be categorised under Juvenile Victims
#   MISSING JUVENILE (19848 Cases) - to be excluded from the analysis

# Subsetting the cases with "CHILD" in the description
sum(grepl("CHILD", d$Descript) == "TRUE") # 8857 cases
CHLD <- d[grep("CHILD", d$Descript)]
summary(CHLD$Descript) # All Child VICTIMS


### --- Victims ---

# Creating a datatable with only cases involving Juvenile victims
sum(grepl("JUVENILE VICTIM", d$Descript) == "TRUE") # 442 cases
vector <- c("JUVENILE VICTIM", "SELLING RESTRICTED GLUE TO JUVENILES", "CHILD")
sum(grepl(paste(vector, collapse = "|") , d$Descript)) # 9304 cases
v <- d[grep(paste(vector, collapse = "|"), d$Descript)]


### --- Perpetrators ---

# Creating a datatable with only cases involving Juvenile perpetrators
exclude <- c("JUVENILE VICTIM", "SELLING RESTRICTED GLUE TO JUVENILES", "MISSING JUVENILE", "CHILD")
p <- JVNL[!grep(paste(exclude, collapse = "|"), JVNL$Descript)] # 3695 cases


# Creating separate CSV files for visualisation on Tableau
fwrite(v, "Victims.csv")
fwrite(p, "Perpetrators.csv")


# ==================== Time-Series Forecasting ====================


### --- All Cases ---

# Format Date column to Datetime
d[, "Date":= as.Date(Date, format = '%m/%d/%Y')]
d[, "YearMonth":= format(Date, "%Y-%m")]
d[, "Year" := format(Date, "%Y")]

# Train-Test Split
trainset <- d[Year != "2017" & Year != "2018"] # Trainset consists of data from 2003-2016
testset <- d[Year == "2017"] # Testset consists of data in 2017
# 2018 data not taken into consideration because incomplete

# Create Time-Series objects for forecasting
train <- trainset[, .N, by = YearMonth][order(YearMonth)]
train.ts <- ts(train$N, frequency = 12, start = c(2003, 1))

test <- testset[, .N, by = YearMonth][order(YearMonth)]
test.ts <- ts(test$N, frequency = 12, start = c(2017, 1))

plot(train.ts) # Looks like an additive model

# Create Hybrid Model & Forecast
train.m <- hybridModel(train.ts)
train.forecast <- forecast(train.m, h = 12)

plot(train.forecast, main = "12-Month Forecast of Criminal Cases in San Francisco", xlab = "Month", ylab = "Number of Cases")
accuracy(train.forecast, test.ts) # Testset RMSE = 479
mean(test.ts) # 12898 cases
# Testset RMSE / 12898 cases = 0.037


### --- Juvenile Victims ---

# Format Date column to Datetime
v[, "Date" := as.Date(Date, format = '%m/%d/%Y')]
v[, "YearMonth" := format(Date, "%Y-%m")]
v[, "Year" := format(Date, "%Y")]

# Train-Test Split
v.trainset <- v[Year != "2017" & Year != "2018"] # Trainset consists of data from 2003-2016
v.testset <- v[Year == "2017"] # Testset consists of data in 2017
# 2018 data not taken into consideration because incomplete

# Create Time Series object for forecasting
v.train <- v.trainset[, .N, by = YearMonth][order(YearMonth)]
v.train.ts <- ts(v.train$N, frequency = 12, start = c(2003, 1))

v.test <- v.testset[, .N, by = YearMonth][order(YearMonth)]
v.test.ts <- ts(v.test$N, frequency = 12, start = c(2017, 1))

plot(v.train.ts) # Looks like an additive model

# Create Hybrid Model & Forecast
v.train.m <- hybridModel(v.train.ts)
v.train.forecast <- forecast(v.train.m, h = 12)

plot(v.train.forecast, main = "12-Month Forecast of Juvenile Victim Cases in San Francisco", xlab = "Month", ylab = "Number of Cases")
accuracy(v.train.forecast, v.test.ts) # Testset RMSE = 9.5
mean(v.test.ts) # 55.3 cases
# Testset RMSE / 12898 cases = 0.17


### --- Juvenile Perpetrators ---

# Format Date column to Datetime
p[, "Date" := as.Date(Date, format = '%m/%d/%Y')]
p[, "YearMonth" := format(Date, "%Y-%m")]
p[, "Year" := format(Date, "%Y")]

# Train-Test Split
p.trainset <- p[Year != "2017" & Year != "2018"] # Trainset consists of data from 2003-2016
p.testset <- p[Year == "2017"] # Testset consists of data in 2017
# 2018 data not taken into consideration because incomplete

# Create Time Series object for forecasting
p.train <- p.trainset[, .N, by = YearMonth][order(YearMonth)]
p.train.ts <- ts(p.train$N, frequency = 12, start = c(2003, 1))

p.test <- p.testset[, .N, by = YearMonth][order(YearMonth)]
p.test.ts <- ts(p.test$N, frequency = 12, start = c(2017, 1))

plot(p.train.ts) # Looks like a additive model

# Create Hybrid Model & Forecast
p.train.m <- hybridModel(p.train.ts)
p.train.forecast <- forecast(p.train.m, h = 12)

plot(p.train.forecast, main = "12-Month Forecast of Juvenile Perpetrator Cases in San Francisco", xlab = "Month", ylab = "Number of Cases")
accuracy(p.train.forecast, p.test.ts) # Testset RMSE = 14.2
mean(p.test.ts) # 52.4 cases
# Testset RMSE / 12898 cases = 0.27


# ============================================================
# Appendix - Additional Variables (Block, Temperature, Weather)
# ============================================================


dt <- fread("SFCrimeData.csv", stringsAsFactors = TRUE)


# ==================== Data Preparation ====================


# Feature Engineering to include Weather & Temperature; Split Time & Date to constituents
dt$TimeHour <- substr(dt$Time,0,2)
dt$TimeMinute <- substr(dt$Time,4,5)
dt$DateMM <- substr(dt$Date,0,2)
dt$DateDD <- substr(dt$Date,4,5)
dt$DateYYYY <- substr(dt$Date,7,10)

# Convert string to Date
dt$Date <- as.Date(dt$Date, format="%m/%d/%Y")

# Find number of rows between 2012-10-1 & 2017-11-30
nrow(dt) #2215024
nrow(subset(subset(dt, Date >= as.Date("2012-10-1")), Date <= as.Date("2017-11-30"))) #789519
preCombined.dt <- subset(subset(dt, Date >= as.Date("2012-10-1")), Date <= as.Date("2017-11-30"))
nrow(preCombined.dt)/nrow(dt) #35.6% of data after 2012-10-1

# Create datetime column to combine with next steps
preCombined.dt <- transform(preCombined.dt, datetime=paste(Date, Time, sep=" "))
preCombined.dt$datetime <- substr(preCombined.dt$datetime,0,14)
preCombined.dt$datetime <- paste(preCombined.dt$datetime, "00:00", sep="")
preCombined.dt$datetime <- as.factor(preCombined.dt$datetime)

# Reading in hourly tempetature & weather data
temp.dt <- fread("temperature.csv", stringsAsFactors = TRUE)
weather.dt <- fread("weather_description.csv", stringsAsFactors = TRUE)
preCombined.temp.dt <- temp.dt[,c("datetime","San Francisco")]
preCombined.weather.dt <- weather.dt[,c("datetime","San Francisco")]

# Merging data based on date time to include hourly weather & temperature
combined1.dt <- merge(preCombined.dt, preCombined.temp.dt, by.x="datetime", by.y="datetime")
combined2.dt <- merge(combined1.dt, preCombined.weather.dt, by.x="datetime", by.y="datetime")

# Removing NA
sum(is.na(combined2.dt))
newData.dt<-na.omit(combined2.dt)
sum(is.na(newData.dt))

# Rename data columns
colnames(newData.dt)[18] <- "Weather"
colnames(newData.dt)[17] <- "Temperature"

# Consolidating weather
newData.dt$WeatherConsolidated <- case_when(str_detect(newData.dt$Weather, 'clouds')|str_detect(newData.dt$Weather, 'proximity')|str_detect(newData.dt$Weather, 'drizzle') ~ 'moderate weather',
                                            str_detect(newData.dt$Weather, 'rain')|str_detect(newData.dt$Weather, 'snow') ~ 'bad weather',
                                            str_detect(newData.dt$Weather, 'clear') ~ 'good weather',
                                            TRUE ~ 'mild weather')
newData.dt$WeatherConsolidated <- factor(newData.dt$WeatherConsolidated, ordered = T, levels = c("good weather", "mild weather", "moderate weather", "bad weather"))

# Convert to string to write to csv
newData.dt$datetime <- as.character(newData.dt$datetime)
newData.dt$Date <- as.character(newData.dt$Date)

# Create new 'block column'
str_detect(newData.dt$Address, 'Block')
newData.dt$Block <- ifelse(str_detect(newData.dt$Address, 'Block')|str_detect(newData.dt$Address, 'block'), TRUE, FALSE)

# 1st round drop - Dropping columns that we will not know beforehand or useless to models
newData.dt[,Descript:=NULL]
newData.dt[,Resolution:=NULL]
newData.dt[,datetime:=NULL]
newData.dt[,Date:=NULL]
newData.dt[,Time:=NULL]
newData.dt[,Address:=NULL]

fwrite(newData.dt, "newData.csv")

# Further preprocessing for bucketing by region
newData.dt <- fread("newData.csv", stringsAsFactors = TRUE)
newData.dt$XRegion <- cut(newData.dt$X, 6)
newData.dt$YRegion <- cut(newData.dt$Y, 8)
summary(newData.dt$XRegion)
summary(newData.dt$YRegion)

newData.dt$Region <- as.character(interaction(newData.dt$XRegion, newData.dt$YRegion,sep=" "))

fwrite(newData.dt, "newDataRegion.csv")


# ==================== Data Visualization ====================


coul <- brewer.pal(5, "Set2") 
dt <- fread("newData.csv", stringsAsFactors = TRUE)

# Temperature vs. Category
dt$Temperature <- dt$Temperature - 273.15
par(mar=c(4,16,4,4))

boxplot(dt$Temperature~ dt$Category, horizontal = TRUE, las=2, col=coul, ylab = "", xlab= "Temperature Celsius")

# Block vs. Category
counts <- table(dt$Block, dt$Category)
barplot(counts, main="% of Crime in Blocks for each Category",
        col=c("darkblue","red"),
        legend = rownames(counts), horiz=TRUE, las=1)

# WeatherConsolidated vs. Category
dt$WeatherConsolidated <- factor(dt$WeatherConsolidated, ordered = T, levels = c("good weather", "mild weather", "moderate weather", "bad weather"))
counts <- table(dt$WeatherConsolidated, dt$Category)
barplot(counts, main="Severity of weather for each Category",
        col=c("darkblue","red", "green", "purple"),
        legend = rownames(counts), horiz=TRUE, las=1)


# ==================== CART ====================


set.seed(2020)
options(digits = 5)

dt <- fread("newDataRegion.csv", stringsAsFactors = TRUE)
summary(dt)

# Drop X and Y
dt[,X:=NULL]
dt[,Y:=NULL]
dt[,XRegion:=NULL]
dt[,YRegion:=NULL]

# Train-Test Split
train <- sample.split(Y = dt$Category, SplitRatio = 0.7)
trainset <- subset(dt, train == T)
testset <- subset(dt, train == F)

summary(trainset)
summary(testset)

# default cp = 0.01. Set cp = 0 to guarantee no pruning.
cart <- rpart(Category ~ ., data = trainset, method = 'anova', control = rpart.control(minsplit = 20000, cp = 0))
printcp(cart)
plotcp(cart)

# Compute min CVerror + 1SE in maximal tree cart1
CVerror.cap <- cart$cptable[which.min(cart$cptable[,"xerror"]), "xerror"] + cart$cptable[which.min(cart$cptable[,"xerror"]), "xstd"]

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree cart.
i <- 1; j<- 4
while (cart$cptable[i,j] > CVerror.cap) { 
  i <- i + 1 
}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp.opt = ifelse(i > 1, sqrt(cart$cptable[i,1] * cart$cptable[i-1,1]), 1)

cart.new <- prune(cart, cp = cp.opt)
rpart.plot(cart.new)

predictedCat<- predict(cart.new, newdata = testset)
tab <- table(predictedCat,testset$Category)
accuracy <- round(mean(predictedCat == testset$Category),3)
accuracy

cart.new$variable.importance


# ==================== K-Nearest-Neighbour Classification ====================


set.seed(2020)
options(digits = 5)

dt <- fread("newData.csv", stringsAsFactors = TRUE)
summary(dt)

summary(dt$Weather)

dt$WeatherConsolidated <- case_when(str_detect(dt$Weather, 'clouds')|str_detect(dt$Weather, 'proximity')|str_detect(dt$Weather, 'drizzle') ~ 2,
                                    str_detect(dt$Weather, 'rain')|str_detect(dt$Weather, 'snow') ~ 3,
                                    str_detect(dt$Weather, 'clear') ~ 0,
                                    TRUE ~ 1) 
summary(dt$WeatherConsolidated)

sampled.dt <- sample_n(dt, 100000)

newSampled.dt <- sampled.dt[,c("Category","Y","X","Temperature","DayOfWeek","TimeHour","WeatherConsolidated","Block")]

newSampled.dt$DayOfWeek <- as.numeric(newSampled.dt$DayOfWeek)
newSampled.dt$TimeHour <- as.numeric(newSampled.dt$TimeHour)
newSampled.dt$WeatherConsolidated <- as.numeric(newSampled.dt$WeatherConsolidated)
newSampled.dt$Block <- as.numeric(newSampled.dt$Block)


# Normalization function is created
cat <- newSampled.dt$Category
nor <-function(x) { (x -min(x))/(max(x)-min(x))   }
newSampled.dt <- as.data.frame(lapply(newSampled.dt[,c("Category","Y","X","Temperature","DayOfWeek","TimeHour","Weather","Block")], nor))
newSampled.dt$Category <- cat

# Train-Test Split
train <- sample.split(Y = newSampled.dt$Category, SplitRatio = 0.7)
trainset <- subset(newSampled.dt, train == T)
testset <- subset(newSampled.dt, train == F)

summary(trainset)
summary(testset)

length(unique(trainset$Category))
sapply(newSampled.dt, class)

pr <- knn(trainset,testset,cl=trainset$Category,k=length(unique(trainset$Category)))
tab <- table(pr,testset$Category)
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)

# 10 cross validation
# trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
# knn_fit <- train(Category ~., data = trainset, method = "knn",
#                  trControl=trctrl,
#                  preProcess = c("center", "scale"),
#                  )
# knn_fit
# test_pred <- predict(knn_fit, newdata = testset)
# confusionMatrix(test_pred, testset$Category )


#======================================================== END =========================================================