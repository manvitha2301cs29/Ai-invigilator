"""
database_PHASE_C.py
---------------------
Postgres engine + session factory, read from DATABASE_URL (see
.env.example_PHASE_C). Uses psycopg (v3), the current recommended
PostgreSQL driver for SQLAlchemy 2.x -- not the older psycopg2, which
is still supported but not the forward-looking choice for a new
project.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://invigilator:dev_password_change_in_production@localhost:5432/study_invigilator",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency -- yields a session, always closes it after
    the request, even if an exception occurred."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables if they don't already exist. Fine for Phase C
    development; Phase G's deployment should switch to real Alembic
    migrations (mentioned in 01_SETUP_ENVIRONMENT.txt) instead of
    relying on create_all for anything beyond local dev, since
    create_all cannot handle schema CHANGES to existing tables, only
    initial creation."""
    import models  # noqa: F401 -- ensures models are registered
                             # on Base.metadata before create_all runs.
                             # Plain (non-relative) import deliberately:
                             # this backend/ folder is run as a script
                             # location (uvicorn main_PHASE_C:app), not
                             # imported as a package, so a relative
                             # "from . import ..." fails with
                             # "attempted relative import with no known
                             # parent package." If you later restructure
                             # backend/ into a proper installed package,
                             # switch this back to a relative import then.
    models.Base.metadata.create_all(bind=engine)
