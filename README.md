# project-NTA
This repository contains the code for our 50.038 project, where we aim to create a model that is able to read new posts to the *r/AmITheAsshole* (AITA) subreddit, and predict the most popular judgement label from the community, even before any votes have been cast.

### visualisation.ipynb
This notebook contains the code for the visualisation done, namely:
* Class Frequencies
* Score Distributions
* Most Informative Words (using sklearn's chi2)

### Doc2Vec(SVM).ipynb
This notebook contains the code for the Doc2Vec model we have implemented on the AITA dataset to learn the feature representations among the documents. It creates document embeddings, which are input into a classifier - we have chosen Support Vector Machine (SVM) as the classification model.

### CNN_oversampling.ipynb & CNN_undersampling.ipynb
These notebook contains the code for the CNN model we have implemented, including a predictor function to predict the classification of an AITA text (Asshole/Not-Asshole) with the corresponding accuracy score.

The former includes imbalanced-learn's *SMOTE* class for oversampling on the under-represented class (is_asshole==1), while the latter includes imbalanced-learn's *NearMiss* class for undersampling on the over-represented class (is_asshole==0).
