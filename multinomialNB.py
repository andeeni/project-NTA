import pandas as pd
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('../data/aita_clean.csv')

# concatenate title and body of each post
df['text'] = df['title'] + df['body'].fillna('')
df = df[df['score'] >= 10]

# cross-validator
# Each fold is then used once as a validation while the k - 1 remaining folds form the training set
# split into 5 datasets, shuffle the data before splitting to each batches
kf = KFold(n_splits=5, random_state=2, shuffle=True)
kf_split = kf.split(df)

# TF-IDF Vectorizer
# input = 'content', encoding = 'latin-1', max_features : ordered by term frequency
# norm = 'l2' -> sum of squares of vector elements is 1
# min-df = 5 ->  ignore terms that have a document frequency strictly lower than
# sublinear_tf = True -> apply sublinear tf scaling
# ngram_range = (1,1) -> unigram
vec = TfidfVectorizer(encoding='latin-1',
                        stop_words='english', 
                        ngram_range=(1,1),
                        min_df=5, 
                        max_features = 10000,
                        sublinear_tf=True)

model = MultinomialNB()

sm = SMOTE(random_state=42) # for class balancing

accuracies = []

for i in range(1,6):
    print("Processing fold #" + str(i))

    result = next(kf_split)
    train = df.iloc[result[0]]
    test = df.iloc[result[1]]

    feats = vec.fit_transform(train.text).toarray() # apply TF-IDF on training set
    labels = train.is_asshole
    
    X_train,y_train = sm.fit_resample(feats,labels) # using smote to resample
    X_test = vec.transform(test.text).toarray() # apply TF-IDF on test set
    y_test = test.is_asshole
    
    model.fit(X_train, y_train)
    accuracies.append( model.score(X_test, y_test) )
    
print(accuracies)

