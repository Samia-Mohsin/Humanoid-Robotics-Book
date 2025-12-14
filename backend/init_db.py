#!/usr/bin/env python3
"""
Script to initialize the database tables
"""
import asyncio
from core.database import engine
from models import Base

async def init_db():
    """Initialize the database by creating all tables"""
    print("Initializing database tables...")

    try:
        # Create all tables defined in the models
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        print("Database tables created successfully!")

        # For verification, we can list the table names from the metadata
        table_names = list(Base.metadata.tables.keys())
        print(f"Created tables: {table_names}")

    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(init_db())