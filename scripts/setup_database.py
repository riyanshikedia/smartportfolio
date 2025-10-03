#!/usr/bin/env python3
"""
Database Setup Script
Creates the SmartPortfolio database and all required tables
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_database_connection():
    """Create connection to SQL Server (master database)"""
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '1433')
    db_user = os.getenv('DB_USER', 'sa')
    db_password = os.getenv('DB_PASSWORD', '')
    
    # Connect to master database first
    connection_string = f"mssql+pymssql://{db_user}:{db_password}@{db_host}:{db_port}/master"
    
    try:
        engine = create_engine(connection_string, echo=False)
        logger.info("✅ Connected to SQL Server")
        return engine
    except Exception as e:
        logger.error(f"❌ Failed to connect to SQL Server: {e}")
        raise

def create_database(engine):
    """Create SmartPortfolio database if it doesn't exist"""
    try:
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM sys.databases 
                WHERE name = 'SmartPortfolio'
            """))
            db_exists = result.scalar() > 0
            
            if not db_exists:
                logger.info("📊 Creating SmartPortfolio database...")
                conn.execute(text("CREATE DATABASE SmartPortfolio"))
                conn.commit()
                logger.info("✅ SmartPortfolio database created")
            else:
                logger.info("✅ SmartPortfolio database already exists")
                
    except Exception as e:
        logger.error(f"❌ Error creating database: {e}")
        raise

def execute_schema_file(engine):
    """Execute the schema.sql file to create all tables"""
    schema_file = project_root / "database" / "schema.sql"
    
    if not schema_file.exists():
        logger.error(f"❌ Schema file not found: {schema_file}")
        return False
    
    try:
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Split by GO statements and execute each batch
        batches = [batch.strip() for batch in schema_sql.split('GO') if batch.strip()]
        
        with engine.connect() as conn:
            for i, batch in enumerate(batches):
                if batch.strip():
                    logger.info(f"Executing batch {i+1}/{len(batches)}...")
                    conn.execute(text(batch))
                    conn.commit()
        
        logger.info("✅ Database schema created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error executing schema: {e}")
        return False

def test_database_setup():
    """Test that all tables were created successfully"""
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '1433')
    db_name = os.getenv('DB_NAME', 'SmartPortfolio')
    db_user = os.getenv('DB_USER', 'sa')
    db_password = os.getenv('DB_PASSWORD', '')
    
    connection_string = f"mssql+pymssql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    try:
        engine = create_engine(connection_string, echo=False)
        
        with engine.connect() as conn:
            # Check if key tables exist
            tables_to_check = [
                'market.sp500_tickers',
                'market.daily_prices',
                'fundamental.financial_ratios',
                'portfolio.positions',
                'ml_models.return_predictions'
            ]
            
            for table in tables_to_check:
                schema, table_name = table.split('.')
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = :schema 
                    AND TABLE_NAME = :table_name
                """), {"schema": schema, "table_name": table_name})
                
                exists = result.scalar() > 0
                if exists:
                    logger.info(f"✅ {table} exists")
                else:
                    logger.error(f"❌ {table} missing")
                    return False
        
        logger.info("🎉 Database setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing database setup: {e}")
        return False

def main():
    """Main execution function"""
    global logger
    logger = setup_logging()
    
    print("=" * 80)
    print("🚀 SMARTPORTFOLIO - DATABASE SETUP")
    print("=" * 80)
    
    try:
        # Check if .env file exists
        env_file = project_root / ".env"
        if not env_file.exists():
            logger.error("❌ .env file not found!")
            logger.info("Please create a .env file with your database credentials:")
            logger.info("DB_HOST=localhost")
            logger.info("DB_PORT=1433")
            logger.info("DB_NAME=SmartPortfolio")
            logger.info("DB_USER=sa")
            logger.info("DB_PASSWORD=YourPassword")
            return 1
        
        # Step 1: Connect to SQL Server
        logger.info("📂 Connecting to SQL Server...")
        engine = create_database_connection()
        
        # Step 2: Create database
        logger.info("📊 Setting up database...")
        create_database(engine)
        
        # Step 3: Execute schema
        logger.info("📋 Creating tables and schemas...")
        if not execute_schema_file(engine):
            return 1
        
        # Step 4: Test setup
        logger.info("🧪 Testing database setup...")
        if not test_database_setup():
            return 1
        
        print("\n" + "=" * 80)
        print("✅ DATABASE SETUP COMPLETE!")
        print("=" * 80)
        print("\n🎯 Next steps:")
        print("1. Run: python scripts/run_data_collection.py")
        print("2. Run: python scripts/run_technical_indicators.py")
        print("3. Continue with other scripts...")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
