USA Road Accidents Severity Prediction





## Introduction



## 1.1  Background



Traffic accidents are a major public health problem in the United States. In 2022, over 42,795 people were killed in traffic accidents and over 4.2 million people were injured[1]. The US Accidents dataset provides a comprehensive overview of traffic accidents in the United States, including information on the location, time, and severity of accidents, as well as the type of vehicles involved, the number of casualties, and the contributing factors. Our dataset contains 46 columns and 7728393 rows.



## 1.2  Motivation



The US Accidents dataset can be used to develop a predictive model for traffic accidents, which could inform a variety of interventions to reduce the number and severity of traffic accidents. Machine learning algorithms have been shown to be effective for predicting traffic accidents, and the US Accidents dataset provides a valuable resource for training and evaluating these algorithms. By developing a predictive model for traffic accidents using the US Accidents dataset, we can help to reduce the number and severity of traffic accidents and save lives.



## 1.3  Goal



The goal of this data science project is to develop a predictive model to identify drivers and vehicles at high risk of being involved in a traffic accident. This model will be developed using the US Accidents dataset and will be used to inform targeted interventions to reduce the number and severity of traffic accidents.



## Methodology



Libraries like Pandas[2], Seaborn[3], Matplotlib, Plotly, and Sci-kit learn[4] were chosen for their robust data manipulation and visualization capabilities. Pandas is essential for handling large datasets efficiently, Seaborn and Matplotlib are excellent for static visualizations, and Plotly offers interactive capabilities. Pandas was used to load and pre-process the data (`pd.read_csv()`). Seaborn and Matplotlib were utilized for creating various charts, and Plotly was employed for more dynamic, interactive visualizations. Feature engineering and Data processing is done before data is split and loaded in the model. To find the optimum model we have done hyperparameter tuning, feature importance, GridSearchCV. Since the data was imbalanced we did oversampling.



## 2.1 Data Pre-processing



We initiated the process by organizing the dataset of US Car accidents. This involved cleaning the text data, handling missing values, and ensuring the data is in a suitable format for analysis, making use of NumPy and Pandas.

 

We checked for null values, unique values in each column, and removed unnecessary columnsID','Weather_Timestamp','Weather_Condition','End_Time','End_Time1','Start_Time1','Start_Time','County','Description','End_Lat','End_Lng','Wind_Chill(F)','Precipitation(in)','Start_Lat','Start_Lng','Airport_Code','Street','Country','State','Zipcode','Country' were excluded as they were unnecessary for our purpose.



The dataset was then split into training and test sets, a critical step in machine learning. By allocating 20% or 25% of the data to the test set, we were able to train the model on a substantial portion of the data while retaining a significant subset for evaluating the model's performance on unseen data.



Label encoding for two categorical columns, namely source and weather bins since they are of categorical values



Boolean mapping of 0 for false and 1 for true is done for boolean columns.



## Feature Engineering



In the development of our machine learning model, we undertook several key steps in feature engineering to prepare the dataset for analysis. Our approach was structured and methodical, ensuring that the data was optimally formatted for the algorithms we intended to use.



Firstly, we created new attributes like day, month, hour, minute, day of the week, year, accident duration, comfort index from the start time & end time.



Next, we focused on defining the target variable, which in our case was 'Severity'. We separated this variable from the rest of the dataset to distinguish between the input features and the output variable that the model was to predict. This separation is a standard practice in supervised learning models.



In terms of feature selection, we excluded the 'Severity' column from the dataset used for training the model. This ensured that the model would not be inadvertently influenced by the target variable during the learning process.



Understanding the importance of feature scaling, especially for certain algorithms, we then standardized the features using the StandardScaler. This process involved removing the mean and scaling the features to unit variance, thus normalizing the data and preventing any feature from disproportionately influencing the model's predictions.

In addition to these steps, we employed the K-Means clustering algorithm to discern inherent groupings within a dataset that appeared to be geospatial in nature. The dataset comprised variables such as start and end coordinates (latitude and longitude), which focused on location data.



We began by pre-processing the data for clustering, selecting 'Start_Lat' and 'Start_Lng' as the primary features. Recognizing the sensitivity of K-Means to the scale of the data, we applied the `StandardScaler` from `sklearn.preprocessing` to normalize these features. This ensured a balanced contribution of each feature to the distance calculations used in the clustering process.



With the data scaled, our next objective was to determine the optimal number of clusters. For this, we utilized the Elbow Method, a widely acknowledged technique for ascertaining the appropriate cluster count. By calculating the within-cluster sum of squares (WCSS) for a range of cluster values — from 1 to 10 — and plotting these values, we were able to visually identify the "elbow" point. This point is characterized by a diminishing rate of decrease in WCSS, signifying that additional clusters would not substantially enhance the model's performance.



Having identified the elbow point, we proceeded with the K-Means clustering using the `sklearn.cluster.KMeans` class. The algorithm was initialized with parameters including the number of clusters, determined from the Elbow Method, and a `random_state` for reproducible results. We also specified `k-means++` for the initialization of centroids to ensure a more effective and reliable convergence by selecting initial centres that are spread out across the data space.



The `.fit_predict()` method of the K-Means object was then invoked to calculate the cluster centres and assign a cluster label to each data point. These labels were integrated into the original DataFrame as a new column, effectively indicating the cluster membership of each observation.



To visually interpret the results, we generated scatter plots with the data points color-coded according to their cluster assignment. This not only provided an intuitive grasp of the spatial distribution of the clusters but also allowed for the inspection of the centroids, which are the axes around which the clusters are formed.



## 

## Exploratory Data Analysis (EDA)



EDA is fundamental to understand the dataset's characteristics and to uncover initial patterns, anomalies, relationships, and insights. It guides the subsequent steps of the project, including feature selection and model building.







The boxplots presented visualize key variables from a U.S. accident prediction dataset, offering a succinct overview of the data's distribution and variability for each feature. 



Geographic coordinates such as Start and End Latitude/Longitude reveal the spatial spread of accidents, while weather-related measurements—Temperature, Wind Chill, Humidity, Pressure, Visibility, Wind Speed, and Precipitation—highlight environmental conditions at the time of the accidents. Temporal variables are evenly spread, indicating a consistent distribution of accidents over days, weeks, months, and hours. Outliers in Distance, Pressure, Wind Speed, and Precipitation suggest exceptional cases that significantly deviate from the norm. These plots are critical for preliminary data analysis, helping to identify trends, patterns, and anomalies that could be influential in predicting accidents across the U.S.









This bar chart details the number of road accident cases categorized by severity over the years 2016 to 2023. There's a clear increasing trend in accident cases, with each year having more cases than the last. The category of severity that has the highest frequency also appears to increase over the years, indicating not only more accidents but potentially more severe ones.







This chart shows the distribution of road accident cases in the top 10 US cities, broken down by severity. It appears that Miami, Houston, and Los Angeles have the highest number of cases. The chart could be used to identify cities with higher accident rates and potentially target those areas for safety improvements.







This is a bar chart with percentages, showing the proportion of road accidents in the top 10 US cities. The cities are ranked from highest to lowest number of cases. Miami leads with a significant margin, indicating a high prevalence of road accidents in that city.





This chart ranks the top 10 US states by the number of road accident cases. California has the most significant share, followed by Florida and Texas, suggesting these states may benefit from increased road safety measures.







This bar chart presents the cities with the least number of road accidents. The scale is much smaller than the previous charts, and all cities show a very low number of cases, which could indicate effective traffic management or lower traffic volumes.





Similar to the cities chart, this bar chart shows the states with the fewest road accidents. The states like South Dakota, Vermont, and Maine have the lowest numbers, which might reflect lower population densities or better road conditions.







The bar chart correlates road accidents with different weather conditions. Clear and cloudy weather lead to more accidents, which may seem counterintuitive, but it could be because these conditions are more common, or possibly because drivers are less cautious when the weather seems favourable.





This chart details the number of road accidents in different highway time zones. It shows that some time zones, like I-95 N, have a higher incidence of accidents, which could be due to higher traffic volumes or problematic road sections.









This bar chart looks at road accidents on specific streets, likely within a particular city or region. The chart shows a relatively even distribution across the locations, with 473-401 Cut-off Rd having a slightly higher number of cases.







This bar chart compares the number of accidents by time of day—day, night, and sunrise/sunset—categorized by severity levels. The majority of accidents occur during the day, with a high number of severe accidents (Severity 2), while night and sunrise/sunset have significantly fewer accidents. This could suggest that the increased visibility during the day does not necessarily reduce the severity of accidents.







The horizontal bar chart shows the number of accidents across different weather conditions, broken down by severity levels. The most accidents occur in clear weather conditions, followed by cloudy, with severe accidents (Severity 2) being the most common in these conditions. Less severe accidents (Severity 1) are the most common in snowy conditions, which might be due to more cautious driving or less traffic.







This set of bar charts breaks down the number of accidents by the presence of different road features such as amenities, bumps, crossings, junctions, railways, stops, traffic calming measures, and traffic signals. For most features, the presence of the feature corresponds to a lower number of accidents, with a significant number of severe accidents occurring in areas without these features. This suggests that these road features might play a role in reducing accidents or their severity.





The pie charts visualize the percentage of accidents that occur in the presence of specific road features. The vast majority of accidents happen where there are no bumps, junctions, or traffic calming measures, which may indicate that such features are effective in preventing accidents. However, a substantial proportion of accidents occur at locations with traffic signals, which could point to high-traffic areas or possible issues with signal timing or compliance.

## 







## Machine Learning Models & Evaluation Metrics



5.1 Naive Bayes: Given the high Dimensionality of the Data and the simplicity and it’s efficiency, we decided to start with Naive Bayes model as our initial Classifier. They were built for multi-class classification problems which suits out use case to predict 4 types of Accident severity. They also operate on the principle of conditional probability, assuming independence between features, which simplest the computation in multiclass scenarios. This makes Naive Bayes a suitable choice for our problem, offering a good balance between the predictive performance and computational efficiency in handling multi-classes.



5.2 KNN : The K-Nearest Neighbours (KNN) classifier is a simple yet effective machine learning algorithm used to predict accident severity.. It classifies each accident severity by analysing the ‘k’ closest data points in our dataset, leveraging the principle that similar accidents tend to have similar severities. KNN's advantage lies in its simplicity and effectiveness, especially in scenarios with clear proximity-based clusters. However, its performance depends significantly on the choice of 'k' and the distance metric used. 



5.3 Logistic Regression: In predicting accident severity, the Logistic Regression model demonstrated exceptional performance, particularly after meticulous feature engineering. This model is adept at handling multiple classes, making it ideal for categorizing accidents into four distinct severity types. It works by predicting the probability of each severity level for a given accident, utilizing a logistic function. The model's ability to provide clear, interpretable results, coupled with its robustness in multi-class classification, made it the best model in our by having the Accuracy score when compared with other models.



5.4 Decision Trees and Random Forest: Decision Trees efficiently segment our accident data into branches, representing different severity levels. They excel in interpretability, showcasing clear paths of decisions based on accident characteristics. However, they can be prone to overfitting, especially with complex or noisy data which happened in our study as well.



5.5 Oversampling:  Oversampling is also done and the models are executed again, to overcome the imbalance data problem.



5.6 Precision: Precision in the context of classification models refers to the ratio of true positive predictions to the total number of positive predictions made. It is a measure of a model's ability to identify only relevant instances. In our project, precision would denote the proportion of correctly predicted accident severities out of all the severities predicted to be of a certain class. High precision indicates a low false positive rate, meaning the model is reliable when it predicts a specific severity level.



5.7 Recall: Recall, also known as sensitivity, measures the proportion of actual positives that were correctly identified by the model. For our road accident severity prediction, recall answers the question: "Out of all the actual accidents of a certain severity, how many did we correctly predict?" This metric is crucial when the cost of missing an actual positive (such as failing to identify a severe accident) is high.



5.8 F1-score: The F1-score is the harmonic mean of precision and recall, providing a single score that balances both the false positives and false negatives. It is particularly useful when the class distribution is imbalanced, as is often the case in accident severity prediction. An F1-score reaches its best value at 1 (perfect precision and recall) and worst at 0, offering a composite measure of the model's accuracy in predicting each class.



5.9 Support: In the context of a classification report, 'support' refers to the number of actual occurrences of the class in the specified dataset. For our project, it would be the count of instances for each severity level in the test dataset. Support gives insight into the reliability of the metrics, as a class with a very low support has less impact on the overall accuracy of the model.



5.10 Test accuracy: Test accuracy is the ratio of correctly predicted observations to the total observations in the test dataset. It provides a general indication of how often the model is correct across all severity levels. For our predictive model, test accuracy would reflect its effectiveness across the unseen data, which is pivotal for gauging its potential real-world application.



5.11 Accuracy: This metric summarizes the performance of a model by indicating the percentage of total correct predictions. In our case, it would represent the overall ability of the accident severity prediction model to correctly predict the severity level across all classes. However, in datasets with imbalanced classes, accuracy alone can be misleading; thus, it is evaluated alongside precision, recall, and the F1-score to provide a more complete picture of the model's performance.



## Results & Analysis



6.1 KNN Model 



The KNN model achieving an accuracy of 94.4% in the context of accident severity prediction. This high level of accuracy implies that the model is very effective in correctly classifying the severity of accidents in most cases. However since the data is not balanced we need to look at the precision and recall scores as well. Here 0 represents Severity 1, 1 represents Severity 2, 2 represents Severity 3, 3 represents Severity 4





precision    recall  f1-score   support



           0       0.71      0.15      0.25       191

           1       0.94      1.00      0.97     23578

           2       0.76      0.03      0.05       458

           3       0.00      0.00      0.00       773



    accuracy                           0.94     25000

   macro avg       0.60      0.29      0.32     25000

weighted avg       0.91      0.94      0.92     25000

## From the classification report above it could be inferred that the recall value were low for the 4 labels, but the label 2 which represents Severity 2, has high precision. Overall, the imbalance in your dataset heavily influences the model's performance. It's adept at predicting the majority class but not the minority classes. From the below figure it could be inferred that 6 is the best value for ‘k’ for our model.







6.2 Logistic Regression



The initial implementation of the Logistic Regression model on our dataset gave us the train  and the test accuracy of the model was 0.9407 and  0.9416, it is clear that the model is overfitting, to overcome this we create a Logistic Regression instance with class weight set to 'balanced' since the data is not balanced. And yet the model has higher test score when compared to train which could be inferred from the below output:- 



Training Accuracy: 0.61272

Test Accuracy: 0.62144



Classification Report (Test Set):

               precision    recall  f1-score   support



           0       0.10      0.76      0.18       191

           1       0.98      0.62      0.76     23578

           2       0.20      0.70      0.31       458

           3       0.05      0.49      0.10       773



    accuracy                           0.62     25000

   macro avg       0.34      0.64      0.34     25000

weighted avg       0.93      0.62      0.73     25000



We decided to find the important features through feature_importances_, since there was a lot of variables present. After filtering and using only the important features followed by hyper parameter tuning and grid search cv. The below are the results.





Best Hyperparameters: {'C': 0.01, 'solver': 'liblinear'}, are the best hyperparameters after evaluation.

Training Accuracy Score: 0.9129866666666666

Test Accuracy Score : 0.91296



Classification Report (Training Set):

               precision    recall  f1-score   support



           0       0.16      0.61      0.26       485

           1       0.97      0.95      0.96     70680

           2       0.27      0.67      0.39      1431

           3       0.15      0.02      0.04      2404



    accuracy                           0.91     75000

   macro avg       0.39      0.56      0.41     75000

weighted avg       0.92      0.91      0.91     75000



Classification Report (Test Set):

               precision    recall  f1-score   support



           0       0.17      0.56      0.27       191

           1       0.97      0.95      0.96     23578

           2       0.26      0.67      0.38       458

           3       0.13      0.02      0.04       773



    accuracy                           0.91     25000

   macro avg       0.38      0.55      0.41     25000

weighted avg       0.92      0.91      0.91     25000



It could be inferred that the that the model has a higher train score than test, and the recall values are good for both training and testing data, only the Severity 4 has low recall and precision. The Severity 2 (label 1) has the best precision and recall when compared to others. Feature Importance was done to find the important features. and the model had good accuracy score for both test and train. The re-call values has improved after using feature importance for the severity 1 and 3.



6.3 Random Forest 



The random forest classifier displayed very high training accuracy score of 0.99 and the test score was 0.94 at the initial execution. It clear the model is overfitting. Below is the classification report and the model has good precision and recall when compared to other models used in the study. The confusion matrix and the classification report is shown below for the given model:- 



Test Accuracy: 0.95008

Training Accuracy: 0.99936















Classification Report (Test Set):

               precision    recall  f1-score   support



           0       0.84      0.34      0.49       191

           1       0.96      1.00      0.98     23578

           2       0.58      0.29      0.39       458

           3       0.62      0.09      0.16       773



    accuracy                           0.95     25000

   macro avg       0.75      0.43      0.50     25000

weighted avg       0.94      0.95      0.94     25000








Our model performs well for severity 2 with precision and recall rates of 0.96 and 1, respectively, reflected in an f1-score of 0.98. This indicates a strong ability to identify and correctly classify instances of severity 2. However, the performance significantly drops for severity 4, where precision dips to 0.62 and recall to a mere 0.16, suggesting that the model struggles to detect and accurately predict instances of this class. The overall accuracy of 0.95 seems impressive but is misleading due to the class imbalance highlighted by the varying 'support' numbers. The macro averages, which consider all classes equally, reveal underlying weaknesses with lower scores across the board, especially a recall of just 0.57, implying that our model does not perform uniformly across different classes. The weighted average is more favourable due to the class imbalance, which biases the model towards the majority class.







Hyperparameter tuning and grid_search_cv was performed to overcome the issue but the random forest model was still overfitting, but its test accuracy increased to 0.9508. During our analysis Best Hyperparameters: {'max_depth': 20, 'n_estimators': 150} were used to fit and train the model.  



We implemented soft voting hoping that it will improve the model accuracy. 





Classification Report:

              precision    recall  f1-score   support



           0       0.90      0.30      0.45       145

           1       0.95      1.00      0.97     18853

           2       0.65      0.14      0.23       355

           3       0.70      0.04      0.08       647



    accuracy                           0.95     20000

   macro avg       0.80      0.37      0.43     20000

weighted avg       0.94      0.95      0.93     20000



Confusion Matrix:

[[   43   100     2     0]

 [    5 18823    18     7]

 [    0   300    50     5]

 [    0   612     7    28]]



Accuracy Score: 0.9472



6.4 Decision Tree



Initially we did accuracy score using Gini and Entropy to find a better classifier method and below are the accuracy scores:



[Decision Tree -- entropy] accuracy_score: 0.947

[Decision Tree -- gini] accuracy_score: 0.948



For above it can be inferred that gini gave a better index, now we do a 5 cross grid csv of the parameter grid to find the best parameters of the model. 







param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

The below is the output for the model after execution and fitting it with the best hyperparameters. 



Tuned Decision Tree Parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 2}



Best score (accuracy) on training data is 0.9446933333333334

Test set accuracy: 0.947

Training set accuracy: 0.952



Confusion Matrix:

 [[  63   112    14     2]

 [   22 23406    81    69]

 [   11   290   126    31]

 [    0   668    21    84]]



Classification Report:

               precision    recall  f1-score   support



           0       0.66      0.33      0.44       191

           1       0.96      0.99      0.97     23578

           2       0.52      0.28      0.36       458

           3       0.45      0.11      0.18       773



    accuracy                           0.95     25000

   macro avg       0.65      0.43      0.49     25000

weighted avg       0.93      0.95      0.93     25000



It could be inferred that the best hyperparameters is entropy, the maximum level of decision tree is 10 and splitting in the internal nodes should also be 10. At the end of our analysis we got the following scores:-

Best score is 0.9446933333333334

Accuracy: 0.947



This provides us high accuracy but given that we are looking at a classification problem we need to look at the best score for training was 0.9446 and the recall values are not better when compared to Logistic Regression, but the F-1 score was better.



6.5 Models performance After oversampling using Smote



The below figure shows the implementation of the Oversampling using smote we could clearly see how imbalance the data was before and after the implementation of oversampling. The Severity 2 constituted to around 90 percent of the dataset.









6.6 Naive Bayes:



The GaussianNB was used for predicting the accident severity, it could be inferred that the model was not good in prediction and has an accuracy of 0.44 only. The precision and recall values are good for all labels except the third one which is Severity-4.







Classification Report (Test Set):

               precision    recall  f1-score   support



           0       0.78      0.72      0.75     23281

           1       0.87      0.09      0.17     23690

           2       0.32      0.79      0.46     23704

           3       0.38      0.19      0.26     23583



    accuracy                           0.45     94258

   macro avg       0.59      0.45      0.41     94258

weighted avg       0.59      0.45      0.41     94258





6.7 Random Forest:



After implementing oversampling on random forest it has good test and train score of 0.98 and 0.99 respectively. The classification report below shows how well it differentiate the true positive and the negatives without any hyperparameter tuning. 







Training Accuracy: 0.9999045173884444

Test Accuracy: 0.9855715164760551





















Conclusion



In conclusion the a detailed analysis was done to predict the accident severity in United States.  Based on the environmental conditions while driving the predictions were done. This was implemented through machine learning classification models such as the Random forest, Decision trees, Logistic regression, KNN and Naive bayes. Hyper parameter tuning, Grid search CV, the accuracy score, as well as the precision and recall from the classification report is used to evaluate he models.



Based on the above analysis it could be inferred that Logistic Regression before oversampling  was the best model for our severity classification based on the evaluation metrics. This achievement not only demonstrates the potential of data-driven approaches in public safety but also provides valuable tools for policy makers and emergency responders to allocate resources more effectively and mitigate risks.



Key findings from our study, such as the critical factors influencing accident severity, offer a deeper understanding of accident dynamics. The use of diverse models allowed us to approach the problem from different angles, revealing the complex nature of road accidents.



However, our work is not without limitations. The models, while effective, showed varying degrees of performance and highlighted the challenges in handling imbalanced data and ensuring model generalization. These areas present opportunities for future research, suggesting a need for more advanced algorithms or hybrid models that can further enhance prediction accuracy and interpretability.



Looking ahead, integrating real-time data analysis and exploring the impact of external factors like weather and traffic patterns could offer new dimensions to this study. Additionally, expanding the scope to include more geographical locations and demographic details would enhance the robustness and applicability of our findings. Principal analysis could have been done given the high dimensionality of data for future analysis.





References



[1] US Accidents Dataset:  

[2] Seaborn Visualization Library: 

[3] Pandas Documentation: 

[4] Scikit-learn for Machine Learning Scikit-learn



