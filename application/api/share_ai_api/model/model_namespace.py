from flask_restplus import Namespace, Resource, reqparse
from psycopg2.extras import RealDictCursor
from psycopg2.errors import UniqueViolation
from ..data import get_db
from flask import current_app as app, jsonify, session

api = Namespace('model', description='Model related API calls')


@api.route('/all')
class Models(Resource):
    @api.doc('Return models')
    def get(self):
        '''Retrieves a Model'''
        parser = reqparse.RequestParser()
        parser.add_argument('name', required=True, help="Name cannot be blank!")
        args = parser.parse_args()
        cursor = get_db().cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT name, CONCAT('/api/model/', name) as model_url  FROM model where name ILIKE %s LIMIT 10 ", (args['name']+'%',))
        result = cursor.fetchall()
        cursor.close()
        #app.logger.info(result)

        return jsonify(result)

@api.route('/all/random')
class ModelsRandom(Resource):
    @api.doc('Return a random listing of 3 models')
    def get(self):
        cursor = get_db().cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT name, CONCAT('/api/model/', name) as model_url FROM model ORDER BY RANDOM() LIMIT 10 ")
        result = cursor.fetchall()
        cursor.close()
        # app.logger.info(result)

        return jsonify(result)


@api.route('/<name>')
@api.param('name', 'The model name')
class Model(Resource):
    @api.doc('Return model')
    def get(self, name):
        '''Retrieves a Model'''
        # parser = reqparse.RequestParser()
        # parser.add_argument('name', required=True, help="Name cannot be blank!")
        # args = parser.parse_args()
        cursor = get_db().cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
        SELECT *,
        CONCAT('organization/', organization_name) as organization_url,
        CONCAT('model/', name) as model_url,
        ARRAY(select CONCAT('model/',name, '/', v.version_id) from version v where v.name = m.name) as versions
        FROM model m where name = %s """
                       , (name,))

        result = cursor.fetchone()
        if result is None:
            return "",404
        result = dict(result)
        cursor.close()

        return jsonify(result)

    @api.doc('Create model')
    def put(self, name):
        '''Creates an Model'''
        parser = reqparse.RequestParser()

        parser.add_argument('name', required=True, help="Model name cannot be blank!")
        parser.add_argument('organization_name', required=True, help="Organization name cannot be blank!")
        parser.add_argument('description', required=True, help="Description cannot be blank!")
        parser.add_argument('implementation_language', required=True, help="Language cannot be blank!")
        parser.add_argument('implementation_framework', required=True, help="Framework cannot be blank!")
        parser.add_argument('model_domain', required=True, help="Model domain cannot be blank!")
        parser.add_argument('model_subdomain', required=True, help="Model subdomain cannot be blank!")
        parser.add_argument('task', required=True, help="Model task cannot be blank!")

        args = parser.parse_args()

        if 'username' not in session:
            return 401, "Unauthorized, must be logged in"



        cursor = get_db().cursor(cursor_factory=RealDictCursor)

        cursor.execute("SELECT affiliate_type from organization_affiliate where username=%s", (session['username'],))

        result = cursor.fetchone()

        if result is None:
            return 'Unauthorized: Must be in organization',401


        insertQuery = "INSERT INTO model (name, organization_name, description, implementation_language,\
                        implementation_framework, model_domain, model_subdomain, task)\
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"
        try:
            cursor.execute(insertQuery, (args['name'], args['organization_name'],
                                                                   args['description'], args['implementation_language'],
                                                                   args['implementation_framework'], args['model_domain'],
                                                                   args['model_subdomain'], args['task']))

            cursor.execute("COMMIT;")
            cursor.close()

        except UniqueViolation as uv:
                return '',400






        return 200


@api.route('/<model_name>/<version_id>')
@api.param('model_name', 'The model name')
@api.param('version_id', 'The version id')
class ModelVersion(Resource):
    @api.doc('Return model version')
    def get(self, model_name, version_id):




        cursor = get_db().cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
        SELECT *
        FROM version where name = '%s' and version_id = '%s' """
                       % (model_name, version_id))
        result = cursor.fetchone()
        if result is None:
            return "", 404

        result = dict(result)

        if 'username' in session:
            cursor.execute("SELECT 1 from transactions where username=%s and version_id=%s and  model_name=%s",
                           (session['username'], version_id, model_name))

            purchased = cursor.fetchone()

            if purchased is None:
                result['role'] = 'buyer'
            else:
                result['role'] = 'bought'

        else:
            result['role'] = 'guest'

        cursor.close()



        return jsonify(result)

@api.route('/<model_name>/<version_id>/purchase')
@api.param('model_name', 'The model name')
@api.param('version_id', 'The version id')
class BuyModelVersion(Resource):
    @api.doc('Purchase model by current logged in user')
    def post(self, model_name, version_id):

        if 'username' not in session:
            return "", 401

        parser = reqparse.RequestParser()

        parser.add_argument('model_name', required=True, help="Model name cannot be blank!")
        parser.add_argument('version_id', required=True, help="version_id name cannot be blank!")

        args = parser.parse_args()

        cursor = get_db().cursor(cursor_factory=RealDictCursor)

        cursor.execute("SELECT 1 from transactions where username=%s and version_id=%s and  model_name=%s",
                       (session['username'], version_id, model_name))

        result = cursor.fetchone()

        if result is not None:
            return "Already Purchased", 400

        cursor.execute("""SELECT * from organization_affiliate oa join model m on oa.organization_name = m.organization_name
                          where oa.username = %s and m.name = %s""", (session['username'], model_name))

        if cursor.fetchone() is not None:
            return "Cannot Purchase, already in organization", 403

        cursor.execute(
            """INSERT INTO transactions (username, version_id, model_name, transaction_timestamp) VALUES (%s, %s, %s, CURRENT_TIMESTAMP);""",
            (session['username'], args['version_id'], args['model_name']))
        cursor.execute("COMMIT;")

        cursor.close()

        return "", 201
