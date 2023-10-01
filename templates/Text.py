import keras
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from text_package.Basefile import FilterData
from text_package.Basefile import TrainingData
from text_package.Basefile import TextAnalysis
from text_package.Basefile import GetFileSummary

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"

from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *

init_notebook_mode(connected=True)

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV

class FileSummary:

    def __init__(self,file_name):

        self.summary = GetFileSummary(csv_file_name=file_name)

    def get_file_summary(self):

        try:
            self.summary.Summaryanalysis()
        except Exception as e:
            return e

            
class WordAnalysis:
    
    def __init__(self,base_file,feature_name):
        
        self.text_ = TextAnalysis(base_file,feature_name)
       

    def curse_word_count(self,file_name,ascending_order):

        try:
            return pd.DataFrame(
                pd.Series(self.text_.count_bad_words(file_name)).value_counts(ascending=ascending_order))
        except Exception as e:
            return e
        
    def emotions_count(self,file_name,delemeter,ascending_order):

        try:
            return pd.DataFrame(
              pd.Series(self.text_.get_emotions(file_name,delemeter)).value_counts(ascending=ascending_order))

        except Exception as e:
            return e
        
    def motivational_word(self,file_name,ascending_order):

        try:
            return pd.DataFrame(
               pd.Series(self.text_.motivational_phrase(file_name)).value_counts(ascending=ascending_order))
        
        except Exception as e:
            return e
        
    def FinancialWords(self,file_name,ascending_order):

        try:
            return pd.DataFrame(
               pd.Series(self.text_.GetFinancialData(file_name)).value_counts(ascending=ascending_order))
        
        except Exception as e:
            return e
        
    def displayplots(self,type,x,y,data_frame=None,color=None,title=None,xlabel=None,ylabel=None):

        try:

            if type == 'bar':

                fig = px.bar(data_frame=data_frame , x=x,y=y , color=color,title=title)
                fig.update_xaxes(title_text=xlabel)
                fig.update_yaxes(title_text=ylabel)
                fig.show()

            elif type=='line':

                fig = px.line(data_frame=data_frame , x=x,y=y , color=color,title=title)
                fig.update_xaxes(title_text=xlabel)
                fig.update_yaxes(title_text=ylabel)
                fig.show()

            elif type == 'scatter':

                fig = px.scatter(data_frame=data_frame , x=x,y=y , color=color,title=title)
                fig.update_yaxes(title_text=xlabel)
                fig.update_yaxes(title_text=ylabel)
                fig.show()

            elif type=='pie':

                fig = px.pie(data_frame=data_frame , names=x , values=y , color=color,title=title)
                fig.update_xaxes(title_text=xlabel)
                fig.update_xaxes(title_text=ylabel)
                fig.show()

        except Exception as e:

            return e
        
class PhraseAnalysis:

    def __init__(self,base_file,modelname=None):
        
        self.phrase = FilterData(base_file,modelname)
        
    def null_values(self,column_name):

        try:
            return self.phrase.check_null_values(column_name)
        except Exception as e:
            return e

    def label_dict(self,label_column,column_name,word):

        try:
            return pd.DataFrame.from_dict(
                self.phrase.get_label_count(label_column,column_name,word),orient='index',columns=['Total Count']
                )
           
        except Exception as e:
            return e
        
    def filter_sentence(self,column_name,word):

        try:
            return pd.DataFrame(
                list(self.phrase.filter_by_word(column_name,word)),columns=['Text']
                )
        
        except Exception as e:
            return e
    def get_sent_length(self,column_name,word):

        try:
            return pd.DataFrame(
                list(self.phrase.get_sentence_length(column_name,word)),columns=['counts']
                )
        
        except Exception as e:
            return e
        
    def displayplots(self,type,data_frame,x,y,color):

        try:

            if type == 'bar':

                fig = px.bar(data_frame=data_frame , x=x,y=y , color=color)
                fig.show()

            elif type=='line':

                fig = px.line(data_frame=data_frame , x=x,y=y , color=color)
                fig.show()

            elif type == 'scatter':

                fig = px.scatter(data_frame=data_frame , x=x,y=y , color=color)
                fig.show()

            elif type=='pie':

                fig = px.pie(data_frame=data_frame , names=x,values=y , color=color)
                fig.show()

        except Exception as e:

            return e
            
        
    def DataEntityClassification(self,text):

        try:
            return self.phrase.data_entity(text)
        except Exception as e:
            return e

    def displayentityplots(self,text):

        try:
            return self.phrase.displayentity(text)
        except Exception as e:
            return e
        
class ClassificationModelTuning:

    def __init__(self,base_file):

        
        self.training_class = TrainingData(base_file)
        self.tree = DecisionTreeClassifier()
        self.neighbors = KNeighborsClassifier()
        self.forest_classifier = RandomForestClassifier()
        self.xgbc = XGBClassifier()

    def get_dependent_feture(self,column_name):

        try:
            return self.training_class.dependent_feature(column_name)
        except Exception as e:
            return e

    def training_corpus(self,column_name,stemming,stopwords):

        try:
            return self.training_class.training_list(column_name,stemming,stopwords)
        except Exception as e:
            return e
        
    def split_training_and_testing(self,x,y,test_size,random_state):

        try:
            return train_test_split(x,y,test_size=test_size,random_state=random_state)
        except Exception as e:
            return e
        
    def bag_of_words(self,max_features,training_data,testing_data):

        try:
            bow = CountVectorizer(max_features=max_features)
            return bow.fit_transform(training_data).toarray() , bow.transform(testing_data).toarray() , bow
        except Exception as e:
            return e
        
    def TFidf_matrix(self,max_features,training_data,testing_data):

        try:
            tfidf = TfidfVectorizer(max_features=max_features)
            return tfidf.fit_transform(training_data).toarray() , tfidf.transform(testing_data).toarray() , tfidf
        except Exception as e:
            return e
        
    def randomforesttuning(self,x_train,y_train,iter,cv,verbose,random_state,n_jobs,random_grid):

        try:
            rf_random = RandomizedSearchCV(estimator=self.forest_classifier, param_distributions = random_grid, 
                                           n_iter=iter, cv=cv, verbose=verbose, 
                                           random_state=random_state, n_jobs=n_jobs)
            rf_random.fit(x_train,y_train)

            return rf_random.best_params_ , rf_random.best_score_ , rf_random.best_estimator_
        
        except Exception as e:
            return e
        
    def decisiontreetuning(self,x_train,y_train,random_grid,iter,cv,verbose,random_state,n_jobs):

        try:
          
            tree = RandomizedSearchCV(estimator=self.tree, param_distributions = random_grid, 
                                           n_iter=iter, cv=cv, verbose=verbose, 
                                           random_state=random_state, n_jobs=n_jobs)
            tree.fit(x_train,y_train)

            return tree.best_params_ , tree.best_score_ , tree.best_estimator_
        
        except Exception as e:
            return e
        
    def KNNClassifiertuning(self,x_train,y_train,random_grid,iter,cv,verbose,random_state,n_jobs):

        try:

            knnclassifier = RandomizedSearchCV(estimator=self.neighbors, param_distributions = random_grid, 
                                           n_iter=iter, cv=cv, verbose=verbose, 
                                           random_state=random_state, n_jobs=n_jobs)
            knnclassifier.fit(x_train,y_train)

            return knnclassifier.best_params_ , knnclassifier.best_score_ , knnclassifier.best_estimator_
        
        except Exception as e:
            return e
        
    def XGBClassifiertuning(self,x_train,y_train,random_grid,iter,cv,verbose,random_state,n_jobs):

        try:

            xgb_classifier = RandomizedSearchCV(estimator=self.xgbc, param_distributions = random_grid, 
                                           n_iter=iter, cv=cv, verbose=verbose, 
                                           random_state=random_state, n_jobs=n_jobs)
            
            xgb_classifier.fit(x_train,y_train)

            return xgb_classifier.best_params_ , xgb_classifier.best_score_ ,xgb_classifier.best_estimator_

        except Exception as e:
            return e
        
    def NN_tuning(self,start,end,min,max,step,layer_acf,output_class,output_acf,loss,metrics,obj,max_trials,dir_name,x_train,y_train,x_test,y_test,epochs):
    
        try:
            def build_model(hp):

                model = keras.Sequential()
                for i in range(hp.Int('num_layers' , start, end)):
                    model.add(keras.layers.Dense(hp.Int('units_'+str(i),min_value = min,max_value = max,step = step),activation=layer_acf))
    
                model.add(keras.layers.Dense(output_class,activation=output_acf))
                model.compile(
                optimizer=keras.optimizers.Adam(hp.Choice('learning_rate' , [1e-2,1e-3,1e-4])),loss = loss,metrics = [metrics])
                return model
            def tuner():

                tuner = kt.RandomSearch(
                        build_model,
                        objective=obj,
                        max_trials=max_trials,
                        project_name = dir_name)
            
                tuner.search(x_train , y_train, epochs=epochs, validation_data=(x_test,y_test))

                return tuner.results_summary()
        
            tuner()
    
        except Exception as e:

            return e

if __name__ == '__main__':

    pass

