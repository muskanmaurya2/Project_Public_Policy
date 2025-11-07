from db import Base, engine  # Import Base and engine from db.py
from sqlalchemy import Column, Integer, String # ORM..

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(100), nullable=False)

# Create tables in the database
Base.metadata.create_all(bind=engine)
