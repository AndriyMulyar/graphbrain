from .api_v1 import blueprint as api_v1_blueprint
from flask import Flask, g
from flask_cors import CORS
from flask_marshmallow import Marshmallow
import psycopg2
from psycopg2 import pool

def create_app(config_filename=None):
    app = Flask(__name__, instance_relative_config=True)
    CORS(app, supports_credentials=True)
    Marshmallow(app)

    app.config.from_pyfile(config_filename)

    #Creates a pool of database connections for threads to pick up
    app.config['postgreSQL_pool'] = psycopg2.pool.SimpleConnectionPool(1, 20,
                                                                       user="graphbrain_api",
                                                                       password="BXh&R76Z7ZJvxg:+L#WxVY#ykK[f3C",
                                                                       host="db", port="5432", database="graphbrain")
    app.register_blueprint(api_v1_blueprint)
    return app

app = create_app(config_filename='flask.cfg')









