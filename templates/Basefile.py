import re
import nltk
import time
import spacy
import cleantext
nltk.download('stopwords')
import pandas as pd
import seaborn as sns
from collections import Counter
from spacy import displacy
from sklearn.preprocessing import LabelEncoder



class Load_file:

    def __init__(self,base_file):

        self.df = pd.read_csv(base_file)
        self.slang_words = list()
        self.slang_words = list()
        self.total_emotions = list()
        self.motivational_words = list()
        self.word_count = list()
        self.emotions_data = dict()
        self.generalsearch = list()

    def removenull(self):

        self.df = self.df.dropna()
        return self.df
       
    def process_data(self,column_name):
    
       self.df_text = self.removenull()
       return [re.sub('[^a-zA-Z]',' ' , message) for message in self.df_text[column_name].str.lower()]
    
    def gentext(self,column_name):

        return [data.replace(',','').strip() for data in self.process_data(column_name)]
    
    def getarr(self,column_name):

        return [data for data in self.gentext(column_name) if len(data) >= 1]
    
    def get_clean_text(self,column_name):

        arr_lst = self.getarr(column_name)
        return [text.split() for text in arr_lst if len(text) != 0]
    
    def get_flat_arr(self,column_name):

        return [data for text in self.get_clean_text(column_name) for data in text]

class GetFileSummary(Load_file):

    def __init__(self,csv_file_name):

        super().__init__(csv_file_name)

    def get_dimensions(self):

        return f'total rows are : {self.df.shape[0]} and total columns are {self.df.shape[1]}'

    def get_dtypes(self):

        return self.df.dtypes

    def getinfo(self):
                                       
        return self.df.info()

    def description(self):

        return self.df.describe()

    def findnull(self):

        return self.df.isnull().sum()

    def countlabels(self):
            
        for cols in self.df.columns:

            if len(self.df[cols].unique()) <=5 :

                yield self.df[cols].value_counts()
    

    def Summaryanalysis(self):

        print('file dimensions \n')
        print(self.get_dimensions())

        time.sleep(0.5)
        
        print('features dtypes \n')
        print(self.get_dtypes())

        time.sleep(0.5)
        
        print('features info \n')
        print(self.getinfo())

        time.sleep(0.5)
        

        print('file description \n')
        print(self.description())

        time.sleep(0.5)
        
        print('null values\n')
        print(self.findnull())

        time.sleep(0.5)
        
        print('total class Count\n')
        for data in self.countlabels():

            print(data)

class TextAnalysis(Load_file):  

    def __init__(self,base_file,column_name):

        super().__init__(base_file)
        self.data = self.get_flat_arr(column_name)
        
    def count_bad_words(self,file_name):

        with open(file_name) as c_words:

            for line in c_words:

                create_line = line.replace(",", "").replace(
                     "\n", "").replace("'", "").strip()

                self.slang_words.append(create_line.lower())

        c_words.close()
        return [word for word in self.data if word in self.slang_words]
       

    def get_emotions(self,file_name,delemeter):
        
        with open(file_name) as emotions_:

            for line in emotions_:
                create_line = line.replace(",", "").replace(
                    "\n", "").replace("'", "").strip()
                word, emotions = create_line.split(delemeter)
                self.emotions_data[word] = emotions
        
        emotions_.close
        return [value for key,value in self.emotions_data.items() if key in self.data]


    def motivational_phrase(self,file_name):
        
        with open(file_name) as phrase_:

            for line in phrase_:

                create_line = line.replace(",", "").replace(
                    "\n", "").replace("'", "").strip()

                self.motivational_words.append(create_line.lower())
        phrase_.close()
        return [word for word in self.data if word in self.motivational_words]
    
    def GetFinancialData(self,file_name):

        with open(file_name) as general_words:

            for line in general_words:

                create_line = line.replace(",", "").replace(
                    "\n", "").replace("'", "").strip()

                self.generalsearch.append(create_line.lower())
        general_words.close()
        return [word for word in self.data if word in self.generalsearch]
    
class FilterData(Load_file):

    def __init__(self, base_file,model_name=None):
        
        super().__init__(base_file)

        self.file = self.df

        if not model_name == None:
            
            self.ner = spacy.load(model_name)

    
    def check_null_values(self,feature_name):
        
        return self.file[self.file[feature_name].str.len() <= 1]
    
    def get_label_count(self,label_col,feature_name,word):
        
       df = self.file[self.file[feature_name].str.contains(word,case=False)]
       return dict(Counter(df[label_col].values))
    
    def filter_by_word(self,feature_name,word):

        sent_df = self.file[self.file[feature_name].str.contains(word,case=False)]
        return [word for word in sent_df[feature_name]]

    def get_sentence_length(self,feature_name,word):
        
       sentence_len_df = self.file[self.file[feature_name].str.contains(word,case=False)]
       return [len(''.join(word.split())) for word in sentence_len_df[feature_name]]
    
    def check_independent_feature(self,label_col):

        sns.countplot(x=self.file[label_col],data=self.file)

    def NameEntity(self,text):

        raw_text = self.ner(text)

        return [(word.text,word.label_) for word in raw_text.ents if len(word) != 0]
    
    def data_entity(self,text):

        entity = [dict(self.NameEntity(text)) for text in text]
        entity = [dict_val for dict_val in entity if len(dict_val) != 0]

        return {k: v for d in entity for k, v in d.items()}

    def displayentity(self,text):
        
        result = self.ner(text)
        displacy.render(result,style='ent',jupyter=True)

class TrainingData(FilterData):

    def __init__(self, base_file):

        super().__init__(base_file)

        self.data = self.df

    def generate_string(self,column_name):

        return [re.sub('[^a-zA-Z]',' ' , message) for message in self.data[column_name].str.lower()]

    def training_list(self,column_name,stemming,stopwords):

        return [cleantext.clean(word,stemming=stemming,stopwords=stopwords) for word in self.generate_string(column_name)]
    
    def dependent_feature(self,dependent_column):

        return self.data[dependent_column]
    
    def label_encoder(self,feature_name):

        return LabelEncoder().fit_transform(y=self.file[feature_name])
    
    
 

if __name__ == "__main__":

    pass



