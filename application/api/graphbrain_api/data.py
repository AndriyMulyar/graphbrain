from flask import current_app, g
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

@contextmanager
def get_conn():
    if 'conn' in g:
        yield g.conn
    g.conn = current_app.config['postgreSQL_pool'].getconn()
    try:
        yield g.conn
    finally:
        current_app.config['postgreSQL_pool'].putconn(g.conn)

#Retrieves a database connection if the current context has not yet opened one
def get_db():
    if 'conn' not in g:
        g.conn = current_app.config['postgreSQL_pool'].getconn()
    return g.conn