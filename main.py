import yaml
from db_utils import RDSDatabaseConnector

def main():
    
    '''
    The main entry point for the command-line interface.
    '''
    
    def read_db_creds():

        '''
        Retrieves yaml credentials of online db and returns them as dictionary.
        Args:
        path (str): The local file path where yaml file with database credentials is saved. 

        Returns:
           credentials dict(str): stores RDS_HOST, RDS_PASSWORD, RDS_USER ,RDS_DATABASE
           RDS_PORT
        '''

        with open(r'credentials.yaml','r') as stream:
            credentials=yaml.safe_load(stream)

        return credentials   
    
    #Intailising and instance of the connection class and connecting to the RDS online fiance DB
    Con=RDSDatabaseConnector(read_db_creds())
    
    #Printing the list of tables
    Con.list_db_tables()

    #saving the table to df
    loan_payments_df=Con.read_rds_table('loan_payments')

    #Saving to excel
    loan_payments_df.to_csv('loan_payments.csv', index=False)  

if __name__ == "__main__":
    
    main()
