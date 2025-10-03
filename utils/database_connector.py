# utils/database_connector.py

import sqlalchemy as sa
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional
import urllib.parse

load_dotenv()

class DatabaseConnector:
    def __init__(self, use_pymssql=True):
        self.use_pymssql = use_pymssql
        self.setup_logging()
        self.setup_connection()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_connection(self):
        """Create SQL Server database connection"""
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '1433')
        db_name = os.getenv('DB_NAME', 'SmartPortfolio')
        db_user = os.getenv('DB_USER', 'sa')
        db_password = os.getenv('DB_PASSWORD', '')
        
        # URL encode the password
        password_encoded = urllib.parse.quote_plus(db_password)
        
        try:
            if self.use_pymssql:
                # Try pymssql first
                self.connection_string = (
                    f"mssql+pymssql://{db_user}:{password_encoded}@{db_host}:{db_port}/{db_name}"
                )
                self.logger.info("Attempting connection with pymssql...")
            else:
                # Fallback to pyodbc
                self.connection_string = (
                    f"mssql+pyodbc://{db_user}:{password_encoded}@{db_host}:{db_port}/{db_name}"
                    f"?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
                )
                self.logger.info("Attempting connection with pyodbc...")
            
            self.engine = create_engine(
                self.connection_string, 
                echo=False,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600    # Recycle connections after 1 hour
            )
            
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            self.logger.info("✅ Database connection established")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect with {'pymssql' if self.use_pymssql else 'pyodbc'}: {e}")
            
            # Try the other method
            if self.use_pymssql:
                self.logger.info("Retrying with pyodbc...")
                self.use_pymssql = False
                self.setup_connection()
            else:
                raise
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 AS test"))
                test = result.scalar()
                if test == 1:
                    self.logger.info("✅ Database connection test passed")
                    return True
        except Exception as e:
            self.logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            self.logger.info(f"✅ Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            self.logger.error(f"❌ Query execution failed: {e}")
            raise
    
    def insert_data(self, df: pd.DataFrame, table_name: str, schema: str = 'dbo', if_exists: str = 'append'):
        """Insert DataFrame into database table"""
        try:
            full_table_name = f"{schema}.{table_name}"
            
            # Use a connection with proper transaction handling
            with self.engine.begin() as conn:
                df.to_sql(
                    table_name, 
                    conn, 
                    schema=schema,
                    if_exists=if_exists, 
                    index=False,
                    method='multi',
                    chunksize=1000
                )
            
            self.logger.info(f"✅ Inserted {len(df)} records into {full_table_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to insert data into {schema}.{table_name}: {e}")
            return False
    
    def table_exists(self, table_name: str, schema: str = 'dbo') -> bool:
        """Check if table exists"""
        query = """
            SELECT COUNT(*) as count
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = :schema 
            AND TABLE_NAME = :table_name
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"schema": schema, "table_name": table_name})
                count = result.scalar()
            exists = count > 0
            self.logger.info(f"Table {schema}.{table_name} exists: {exists}")
            return exists
        except Exception as e:
            self.logger.error(f"❌ Error checking table existence: {e}")
            return False
    
    def get_latest_date(self, table_name: str, schema: str = 'market', date_column: str = 'date') -> str:
        """Get the latest date in a table"""
        query = f"SELECT MAX({date_column}) as max_date FROM {schema}.{table_name}"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                latest_date = result.scalar()
            return str(latest_date) if latest_date else None
        except Exception as e:
            self.logger.error(f"❌ Error getting latest date: {e}")
            return None
    
    def get_row_count(self, table_name: str, schema: str = 'market') -> int:
        """Get total row count in a table"""
        query = f"SELECT COUNT(*) as count FROM {schema}.{table_name}"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                count = result.scalar()
            return count
        except Exception as e:
            self.logger.error(f"❌ Error getting row count: {e}")
            return 0
    
    def truncate_table(self, table_name: str, schema: str = 'market'):
        """Truncate (clear all data from) a table"""
        try:
            with self.engine.begin() as conn:
                # Use DELETE instead of TRUNCATE to work with foreign keys
                query = f"DELETE FROM {schema}.{table_name}"
                conn.execute(text(query))
            self.logger.info(f"✅ Cleared all data from {schema}.{table_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error truncating {schema}.{table_name}: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        try:
            if self.session:
                self.session.close()
            if self.engine:
                self.engine.dispose()
            self.logger.info("🔌 Database connection closed")
        except Exception as e:
            self.logger.error(f"❌ Error closing connection: {e}")

# Test the connection when module is run directly
if __name__ == "__main__":
    db = DatabaseConnector()
    if db.test_connection():
        print("✅ Database connector is working!")
    else:
        print("❌ Database connector failed!")
    db.close()