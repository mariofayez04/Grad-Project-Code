"""TEST_DATABASE_URL=postgresql://... runs the tests on Postgres instead of SQLite.
Use a throwaway database: every test wipes it."""
import os

import pytest

from src import db


@pytest.fixture(autouse=True)
def postgres_instead_of_sqlite(monkeypatch):
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        monkeypatch.setenv("DATABASE_URL", url)
        with db.connect() as con:
            con.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
