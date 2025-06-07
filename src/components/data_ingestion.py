import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import  train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):  #write code to read from database or files or from api.
        logging.info('Entered the data ingestion method or component')
        try:
            df=pd.read_csv('notebook\data\stud.csv')   # reading data from file. here you can read data from mongodb client or SQL client
            logging.info('Read the dataset as a dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #check directory(artifacts) of train_path exists, if it doesn't exists create one
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #saved imported df in raw_data_path
            
            logging.info('Train test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) #Split raw data into train and test
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # saving data to the path
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            
            logging.info('Ingestion of the data is completed')  
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)