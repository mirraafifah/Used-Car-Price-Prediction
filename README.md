<h1>USED CAR PRICE PREDICTION </h1>

Data source: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho/data

<h2>PROJECT INTRODUCTION:</h2>
Due to the increasing digitization and online market, the used car market receives significant growth. Some companies build applications to make buying and selling used cars easier and build used automotive marketplace so that buyers and sellers easily in making transactions.

<h2>GOAL:</h2>
Increase the growth of used car sales by predicting used car price to customers.
<h2>OBJECTIVE:</h2>
Develop machine learning model that can predict used car price based on related features.
<h2>EVALUATION METRICS:</h2>
RMSE
R2

<h2>SUMMARY EDA:</h2>
1. The distribution of each numerical variables have skewed distribution.
Year: negative skewed
selling price : positive skewed
km_driven :positive skewed
2. Based on ' Selling_price' tren year to year, the newer of year production shows higher price than the older one.
3. The lower the km_driven, the higher the selling price.
4. Used car with diesel fuel has highest average selling price than others
Used car from Trustmark Dealer has highest average selling price than others.
Automatic used car has highest average selling price than others.
Used car from Test Drive Car has highest average selling price than others. Suprisungly it has higher average selling price than first owner.

<h2>SUMMARY PREPROCESSING:</h2>
1. Duplicated values:
   763 entries were duplicated. Since the dataset was collected from internet source and each data does not have an identifier,  we can consider to drop the same value (duplicated value). We still keep the original data in df and copy the removed duplicated data in df2
2. Missing values:
   There are no missing values in dataset
3. Outliers
   The numerical data has some outliers, log transformation was used to handle the outliers number.

<h2>SUMMARY FEATURE ENGINEERING:</h2>
1. Feature Selection
   'name' of car were unutilized since it has too many unique. 
2. Feature Extraction
   'Age' of car was extracte
3. Feature transformation
   Since the categorical data are mostly nominal not an ordinal type, one hot encoding was applied than label encoding.
   Logaritmic transformation was done before applied scaling (minmaxscaler) for numerical data to set the minimum is 0 and the maximum number is 1.
   
<h2>SUMMARY MODEL TRAINING:</h2>
1. Train test split was used to split data with testsize 0.2
2. trained model: linear, ridge, laso, gradient boosting regressor, decision tree regressor, random forest regressor, and XGBoost regressor.
3. Three best score (based on RMSE and R2) model from linear, ridge and gradient boosting (the results were quite similar each other)


<h2>SUMMARY MODEL EVALUATION:</h2>
1. Three best score model were evaluated on data train and data test to observe the overfitting
2. Gradient Boosting regressor was chosen as the best model since the RMSE score from GB is the lowest and the R2 is the highest
3. Hyperparameter tuning was applied to GB model
4. MAPE score was evaluated from the tuned model. The result showed the MAPE score is 0.149 (14.9%)


<h2>SUMMARY MODEL DEPLOYMENT:</h2>
The model was deployed in web app using Streamlit and was run by the app locally. 




