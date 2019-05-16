from flask import current_app, g

#Retrieves a database connection if the current context has not yet opened one
def get_db():
    if 'db' not in g:
        g.db = current_app.config['postgreSQL_pool'].getconn()
    return g.db