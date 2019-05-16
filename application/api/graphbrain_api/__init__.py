from .api_v1 import blueprint as api_v1_blueprint
from flask import Flask, g
from flask_cors import CORS
import psycopg2
from psycopg2 import pool

def create_app(config_filename=None):
    app = Flask(__name__, instance_relative_config=True)
    CORS(app, supports_credentials=True)

    app.config.from_pyfile(config_filename)

    #Creates a pool of database connections for threads to pick up
    app.config['postgreSQL_pool'] = psycopg2.pool.SimpleConnectionPool(1, 20,
                                                                       user="graphbrain_api",
                                                                       password="BXh&R76Z7ZJvxg:+L#WxVY#ykK[f3C",
                                                                       host="db", port="5432", database="graphbrain")
    app.register_blueprint(api_v1_blueprint)
    return app

app = create_app(config_filename='flask.cfg')


#Kills database connections when context dies
@app.teardown_appcontext
def close_conn(e):
    db = g.pop('db', None)
    if db is not None:
        app.config['postgreSQL_pool'].putconn(db)



#An excellent tutorial of the Flask-restplus API
#https://flask-restplus.readthedocs.io/en/stable/scaling.html







