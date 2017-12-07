# Human Activity Recognition Machine Learning Assignment
Geoff Skellams  
5 December 2017  



## Synopsis
The increased adoption of wearable computers and fitness trackers has led to a boon for researchers looking into human movement. This paper explores data captured from a set of weightlifters, who wore fitness trackers and had their movements tracked while performing a range of weightlifting exercises. Each activity in the training set was categorised into one of five separate classses, indicating whether the activity was performed correctly or not. By using a range of machine learning techniques, we explore whether the computer can determine which class each exercise belongs in, from the data alone. We then go on to predict the classes of a much smaller test dataset.

## Data Exploration

Before the data can be analysed, it must first be loaded into memory. If the file do not currently exist in the working directory, they will be automatically downloaded.

Initial exploratory data analysis showed that many of the columns were filled with either blanks, NA values or divide by zero errors. In order to simplify the data cleansing process, all of these values were converted to NA during the file read operation.


```r
# download the training and testing files, if required.
if (!file.exists("pml-training.csv"))
{
    url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(url, dest = "./pml-training.csv", mode = "w") 
    
}

if (!file.exists("pml-testing.csv"))
{
    url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(url, dest = "./pml-testing.csv", mode = "w") 
    
}

# read the training data into a dataframe. Set as NA anything marked as blank, "NA", or a division by zero error
fullTrainingSet <- read.csv("pml-training.csv", na.strings = c("", "NA","#DIV/0!"))
fullTestingSet <- read.csv("pml-testing.csv", na.strings = c("", "NA", "#DIV/0!")) 

dim(fullTrainingSet)
```

```
## [1] 19622   160
```

```r
dim(fullTestingSet)
```

```
## [1]  20 160
```

## Data Analysis

### Utility Functions
To simplify the code structure, much of the data cleansing code was written as two separate functions. 

The first function, **rejectOnNAPercentage()** takes in a single column from a data set and calculates the percentage of values in that column that are NA. If that value is greater than or equal to a threshold percentage passed into the function, it will return TRUE, indicating that the column should be rejected


```r
#---------------------------------------------------------------------------------------------
# Function Name:    rejectOnNAPercentage()
# Function Summary: Returns true if the percentage of values in a column which are NA is above
#                   the threshold value
#---------------------------------------------------------------------------------------------
rejectOnNAPercentage <- function(dataColumn, thresholdVal)
{
    naPercentage <- sum(is.na(dataColumn))/length(dataColumn)
    retValue <- FALSE
    if (naPercentage >= thresholdVal)
    {
        retValue <- TRUE
    }
    retValue
}
```

The other function, **getColsToRemove()**, dramatically reduces the number of variables in the dataset to make the machine learning process easier. Three criteria were used to remove columns from the dataset:

 1. The first seven columns in each row form a unique identifier for that row, detailing who the participant was, and timestamps for when the exercise was performed. These values are irrelevant for the machine learning process, so they will all be removed.
 2. As previously mentioned, there are a large number of columns which are mostly comprised of NA values. Any column which has more than 50% NA values will be removed from the dataset; and
 3. Some columns have next to no variance in its values. These columns provide no real benefit to the machine learning process, so these columns will also be removed.


```r
#---------------------------------------------------------------------------------------------
# Function Name:    getColsToRemove()
# Function Summary: This function will returning a list of column names to be
#                   removed from a dataset in order to make it useful for the machine learning
#                   algorithm.
#---------------------------------------------------------------------------------------------
getColsToRemove <- function(dataset)
{
    # drop the first seven columns because they only contain information about the subject and 
    # timestamps, which are not relevant to the learning process
    colsToRemove <- colnames(dataset)[1:7]
    
    # remove any column where the NA percentage is more than 50%
    highNACols <- sapply(dataset, rejectOnNAPercentage, thresholdVal = 0.5)
    highNAColNames <- colnames(dataset)[highNACols]
    colsToRemove <- append(colsToRemove, highNAColNames)
    
    # remove any column with low variance
    lowVar <- nearZeroVar(dataset, saveMetrics = TRUE)
    lowVarNames <- colnames(dataset)[lowVar$nzv]
    colsToRemove <- append(colsToRemove, lowVarNames)
    
    # return the unique list of column names
    unique(colsToRemove)
}
```

### Data Cleansing and Preparation

Before we can train the machine learning algorithms on the data, we need to clean the full training and testing datasets, removing the unnessary columns. The same set of columns are removed from both the training and test datasets, to ensure that the prediction on the test set will run smoothly. As can be seen, this reduces the number of columns from 160 to only 53.


```r
set.seed(5122017)

# get the list of columns to remove
colNamesToRemove <- getColsToRemove(fullTrainingSet)

# work out which columns to leave in
colsToRemove <- colnames(fullTrainingSet) %in% colNamesToRemove

# remove those columns from the training and test sets
cleanTrainingSet <- fullTrainingSet[, !colsToRemove]
cleanTestSet <- fullTestingSet[, !colsToRemove]

dim(cleanTrainingSet)
```

```
## [1] 19622    53
```

```r
dim(cleanTestSet)
```

```
## [1] 20 53
```

### Data Partitioning

One final step needs to be performed before training the algorithms: partitioning the full training dataset into a training set and a test set, which can be used for cross-validation and error rate measuring. The training dataset will be 80% of the full training set, while the test set will be the remaining 20%


```r
# partition the training data into a training (80%) and test (20%) sets
inTrain <- createDataPartition(y = cleanTrainingSet$classe, p = 0.8, list = FALSE)
trainingData <- cleanTrainingSet[inTrain,]
testingData <- cleanTrainingSet[-inTrain,]

dim(trainingData)
```

```
## [1] 15699    53
```

```r
dim(testingData)
```

```
## [1] 3923   53
```

## Machine Learning

Three different machine learning algorithms were trained with the training dataset, and then validated with the testing set. These algorithms are:

 1. Decision Tree
 2. Boosting
 3. Random Forest

In each case, the code performs the training with the training set, using the **Classe** variable as the outcome, and the other variables as the predictor. The code then performs a prediction using the testing set. It will then show the confusion matrix for the predicted data, while shows how accurate the algorithm was at predicting the class of each observation, compared with the actual recorded class.

### Decision Tree


```r
decTreeModel <- train(classe ~ ., data = trainingData, method = "rpart")
decTreePred <- predict(decTreeModel, newdata = testingData)
confusionMatrix(decTreePred, testingData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1021  321  315  285  109
##          B   13  265   14  118  111
##          C   80  173  355  240  193
##          D    0    0    0    0    0
##          E    2    0    0    0  308
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4968         
##                  95% CI : (0.481, 0.5126)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3421         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9149  0.34914  0.51901   0.0000  0.42718
## Specificity            0.6331  0.91909  0.78821   1.0000  0.99938
## Pos Pred Value         0.4978  0.50864  0.34102      NaN  0.99355
## Neg Pred Value         0.9493  0.85479  0.88584   0.8361  0.88569
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2603  0.06755  0.09049   0.0000  0.07851
## Detection Prevalence   0.5228  0.13281  0.26536   0.0000  0.07902
## Balanced Accuracy      0.7740  0.63412  0.65361   0.5000  0.71328
```

The Decision Tree algorithm was not particularly effective at predicting the correct outcome after training. As can be seen from the confusion matrix, the prediction accuracy was only about 50%. 

The Kappa value is particularly low at 0.28, indicating that the algorithm is only approximately 28% better than random chance at correctly categorising each observation.

### Boosting

```r
boostModel <- train(classe ~ ., data = trainingData, method = "gbm", verbose = FALSE)
boostPred <- predict(boostModel, newdata = testingData)
confusionMatrix(boostPred, testingData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1099   19    0    1    1
##          B   12  711   16    2    9
##          C    2   23  658   22    8
##          D    3    4    9  614    7
##          E    0    2    1    4  696
## 
## Overall Statistics
##                                           
##                Accuracy : 0.963           
##                  95% CI : (0.9567, 0.9687)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9532          
##  Mcnemar's Test P-Value : 0.008519        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9848   0.9368   0.9620   0.9549   0.9653
## Specificity            0.9925   0.9877   0.9830   0.9930   0.9978
## Pos Pred Value         0.9813   0.9480   0.9229   0.9639   0.9900
## Neg Pred Value         0.9939   0.9849   0.9919   0.9912   0.9922
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2801   0.1812   0.1677   0.1565   0.1774
## Detection Prevalence   0.2855   0.1912   0.1817   0.1624   0.1792
## Balanced Accuracy      0.9886   0.9622   0.9725   0.9739   0.9816
```

The Boosting algorithm was *far* more effective than the decision tree in correctly predicting the class variable from the data, getting the correct answer in 96% of cases.

Boosting's Kappa value was 95.3%, which is very close to the model's accuracy, indicating that it will successfully predict the outcome in the vast majority of cases with a high degree of confidence.

### Random Forest


```r
rdmForestModel <- train(classe ~ ., data = trainingData, method = "rf")
rdmForestPred <- predict(rdmForestModel, newdata = testingData)
confusionMatrix(rdmForestPred, testingData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    7    0    0    0
##          B    0  751    0    0    0
##          C    0    1  682    9    0
##          D    0    0    2  634    1
##          E    1    0    0    0  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9946          
##                  95% CI : (0.9918, 0.9967)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9932          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9895   0.9971   0.9860   0.9986
## Specificity            0.9975   1.0000   0.9969   0.9991   0.9997
## Pos Pred Value         0.9938   1.0000   0.9855   0.9953   0.9986
## Neg Pred Value         0.9996   0.9975   0.9994   0.9973   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1914   0.1738   0.1616   0.1835
## Detection Prevalence   0.2860   0.1914   0.1764   0.1624   0.1838
## Balanced Accuracy      0.9983   0.9947   0.9970   0.9925   0.9992
```

The Random Forest algorithm was the most successful of the three algorithms used, correctly predicting the outcome in 99.4% of cases.

Random Forest's Kappa value was 99.2%, indicating that it too is very successful when compared with random chance prediction.

## Final Prediction

Because the Random Forest algorithm had the highest accuracy rating of the three machine learning algorithms tested, we used it first to perform the prediction on the real testing dataset:


```r
cleanRdmForestPred <- predict(rdmForestModel, newdata = cleanTestSet)
cleanRdmForestPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

As a secondary form of validation, we also predicted the test set's classes using the Boosting model:


```r
cleanBoostPred <- predict(boostModel, newdata = cleanTestSet)
cleanBoostPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Both algorithms predicted the test data in exactly the same way, so we can be reasonably confident that the predictions are correct.

## Reference

For more information about the training and testing dataset, please see:

> Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. **Qualitative Activity Recognition of Weight Lifting
> Exercises.** Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart,
> Germany: ACM SIGCHI, 2013.
