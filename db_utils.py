import yaml
import pandas as pd

class RDSDatabaseConnector:

    '''
    Manages connections to online and local PostgreSQL databases and provides 
    methods to list tables and upload data to the database.
    '''
    

    def __init__(self,rds_db_creds):
        
        '''
        Initializes the DatabaseConnector class by setting up the local database engine.

        Args:
          rds_db_creds Dictionary(str): dictionary containing credentials for RDS database with fianance data
        '''

        self.rds_db_creds=rds_db_creds

        self.init_db_engine()
 
    
    def init_db_engine(self):

        '''
        Initializes and returns an SQLAlchemy engine for the online RDS database containg fiance data.

        Returns:
           engine (sqlalchemy.engine.Engine): The SQLAlchemy engine for the online PostgreSQL database.
        '''
        
        #engine from the online db to extract data from
        from sqlalchemy import create_engine
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = self.rds_db_creds['RDS_HOST']
        USER = self.rds_db_creds['RDS_USER']
        PASSWORD = self.rds_db_creds['RDS_PASSWORD']
        DATABASE = self.rds_db_creds['RDS_DATABASE']
        PORT = self.rds_db_creds['RDS_PORT']

        #creating the sqlalchemy engine
        self.engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
        
        

    
    def list_db_tables(self):

        '''
        Retrieves and prints the table names from the database engine and returns them as a list.

        Returns:
           list: A list of table names (str) available in the database.
        '''

        from sqlalchemy import inspect

        table_list = inspect(self.engine).get_table_names()

        print('The table names in the db are:',*table_list)

        return table_list
    
    def read_rds_table(self,table_name):

        '''
        Extracts data from the specified table in the online RDS database
        and returns it as a pandas DataFrame.

        Args:
           table_name (str): The name of the table to read from the database.

        Returns:
           pd.DataFrame: A pandas DataFrame containing the extracted data from the SQL table.
        '''       
      
        # Read the SQL table into a pandas DataFrame
        df = pd.read_sql_table(table_name, con=self.engine)

        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns

        return df 

   
