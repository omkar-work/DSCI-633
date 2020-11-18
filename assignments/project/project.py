import pandas as pd
import time
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def clean_data(data):
    fake = data[data.fraudulent == 1]
    real = data[data.fraudulent == 0]


    num_real, num_fake = data.fraudulent.value_counts()

    real_undersampled = real.sample(num_fake)
    test_undersampled = pd.concat([real_undersampled, fake], axis=0)

    #Y=test_under['fraudulent']
    data_cleaned=test_undersampled.dropna()
    data_cleaned.reset_index(inplace=True)
    test_undersampled['information']= test_undersampled['title']+' '+test_undersampled['location']+' '+test_undersampled['description']+' '+test_undersampled['requirements']
    data=test_undersampled.drop(['title','description','requirements','location'],axis=1)
    data=data.drop('has_questions',axis=1)
    data['has_company_logo'].replace({1:"true",0:"false"},inplace=True)
    data['fraudulent'].replace({0:"Real",1:"Fake"},inplace=True)
    
    
    stopwords = gensim.parsing.preprocessing.STOPWORDS
    
    
    spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–","•","$"]

    for char in spec_chars:
        data['information'] = data['information'].str.replace(char, ' ')
        data['information'] = data['information'].str.split().str.join(" ")
      
    
    data['information']=data['information']+' '+data['has_company_logo']
    data=data.drop('has_company_logo',axis=1)
    data=data.drop('telecommuting',axis=1)
    
    #remove all words like 'in','a','the' from the 'information' column, as those words do not add any significant value to the data.
    data['information'] = data['information'].apply(lambda p: " ".join([w for w in str(p).lower().split() if w not in stopwords]))
    
    return data

class my_model():
    
    def fit(self, X, y):
        # do not exceed 29 mins
        self.classes = y
        
        x_train,x_test,y_train,y_test=train_test_split(X['information'],y,test_size=0.2,random_state=7,shuffle=True)
        #set_trace()
        tfidf=TfidfVectorizer(stop_words='english')
        training=tfidf.fit_transform(x_train.astype('U'))
    
        testing=tfidf.transform(x_test.astype('U'))
        tree=DecisionTreeClassifier(max_depth = 25) 
        rfc = RandomForestClassifier(max_features = 1)
    
                
        rf_grid = {"max_depth": [10,15,25,35,45,50],
                   "criterion": ['gini','entropy'],
                   "min_samples_split": [2,3,4,5],
                   "n_estimators": [10,50,100,150,200]
                  }
        
        
        rf_model = RandomizedSearchCV(rfc,rf_grid,random_state=0,verbose = 5, cv = 10)
        search = rf_model.fit(training, y_train) 
        predictions = search.predict(testing)
        # grid = GridSearchCV(rfc, rf_grid, verbose = 5)
#         print("Grid search : ")
#         gridSearch = grid.fit(training, y_train)
#         pred = gridSearch.predict(testing)
        # print("Randomized : ")
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        x_train,x_test,y_train,y_test=train_test_split(X['information'],self.classes,test_size=0.3,random_state=7,shuffle=True)
        #set_trace()
        tfidf=TfidfVectorizer(stop_words='english')
        training=tfidf.fit_transform(x_train.astype('U'))
    
    
    	#change to unicode.
        testing=tfidf.transform(x_test.astype('U'))
        #tree=DecisionTreeClassifier(max_depth = 25) 
        
        #instance of model.
        rfc = RandomForestClassifier()
    
            
        #range of parameter values to fit on with crossval. 
        rf_grid = {"max_depth": [10,15,25,35,45],
                   "criterion": ['gini','entropy'],
                   "min_samples_split": [2,3,4,5],
                   "n_estimators": [100,150,200]
                  }
        
        # grid = GridSearchCV(rfc, rf_grid, verbose = 5)
#         print("Grid search : ")
#         gridSearch = grid.fit(training, y_train)
#         pred = gridSearch.predict(testing)
#         
#         print(metrics.classification_report(y_test, pred))
#         return pred

		#perform RandomizedSearchCV with the RandomForestClassifier estimator and range of parameters defined above.
        rf_model = RandomizedSearchCV(rfc,rf_grid,random_state=0, n_jobs = -1)
        search = rf_model.fit(training, y_train) 
        print(search.best_params_)
        predictions = search.predict(testing)
        print(metrics.classification_report(y_test, predictions))
        return predictions

if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    data = clean_data(data)
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
#     # Train model
    clf = my_model()
    clf.fit(X, y)
    
    
    predictions = clf.predict(X)
    print(predictions)
    runtime = (time.time() - start) / 60.0
    print("Time taken : ")
    print(runtime)
    