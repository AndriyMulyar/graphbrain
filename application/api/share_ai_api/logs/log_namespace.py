from flask_restplus import Namespace, Resource, reqparse
from psycopg2.extras import RealDictCursor
from psycopg2.errors import UniqueViolation
from ..data import get_db
from flask import current_app as app, jsonify, session

api = Namespace('logs', description='Log related API calls')


@api.route('/')
class Models(Resource):
    @api.doc('Return logs')
    def get(self):
        '''Retrieves logs'''
        cursor = get_db().cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM account_log ORDER BY log_timestamp DESC")
        result = cursor.fetchall()
        cursor.close()
        #app.logger.info(dict(result))

        return jsonify(result)