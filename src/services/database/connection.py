"""
Database connection module for Fraud Detection System.
Provides SQLAlchemy engine and connection utilities.
"""
import os
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_database_uri():
    """
    Get database URI from environment or use default.
    
    Returns:
        str: PostgreSQL connection URI
    """
    return os.getenv(
        'DATABASE_URL',
        'postgresql://fraud_user:fraud_password@localhost:5432/fraud_detection'
    )


def create_db_engine(pool_size=5, max_overflow=10):
    """
    Create SQLAlchemy engine with connection pooling.
    
    Args:
        pool_size (int): Number of connections to maintain in pool
        max_overflow (int): Maximum overflow connections
        
    Returns:
        Engine: SQLAlchemy engine instance
    """
    uri = get_database_uri()
    
    try:
        engine = create_engine(
            uri,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set to True for SQL logging
        )
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logger.info(f"✅ Database connection established: {uri.split('@')[1]}")
        return engine
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        raise


def get_engine():
    """
    Get or create singleton engine instance.
    
    Returns:
        Engine: SQLAlchemy engine
    """
    if not hasattr(get_engine, '_engine'):
        get_engine._engine = create_db_engine()
    return get_engine._engine


def test_connection():
    """
    Test database connection and print schema info.
    
    Returns:
        bool: True if connection successful
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Test query
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"📊 PostgreSQL version: {version}")
            
            # List tables
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            logger.info(f"📋 Tables found: {', '.join(tables)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test module
    print("Testing database connection...")
    if test_connection():
        print("✅ Database connection successful!")
    else:
        print("❌ Database connection failed!")
