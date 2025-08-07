"""
Database configuration for the QA application.

This module defines the SQLAlchemy engine, session and base class used
throughout the application. SQLite is used as the default local
database. If desired, the connection string can be swapped out for a
different relational database without changing the rest of the code.

The `SessionLocal` object is used as a dependency in the FastAPI
routes to provide a database session. It is configured with
`autocommit=False` and `autoflush=False` so changes are only
committed explicitly.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# SQLite URL. Using a relative path will create the file in the
# project directory. If you wish to change where the database lives
# simply edit this URL. Do not include `check_same_thread` when
# connecting to databases other than SQLite.
SQLALCHEMY_DATABASE_URL = "sqlite:///./data.db"

# When connecting to SQLite the `check_same_thread` argument must be
# disabled to allow the connection to be used in different threads.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a configured "Session" class and a session factory. Each
# request in FastAPI will get its own session.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative class definitions.
Base = declarative_base()
