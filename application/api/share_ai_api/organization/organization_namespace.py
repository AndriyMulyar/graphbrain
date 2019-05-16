from flask_restplus import Namespace, Resource, reqparse
from psycopg2.extras import RealDictCursor
from psycopg2.errors import UniqueViolation,ForeignKeyViolation
from ..data import get_db
from flask import jsonify, session
from flask import current_app as app

api = Namespace('organization', description='Organization related API calls')



@api.route('/<organization>')
@api.param('organization', 'The organization name')
class Organization(Resource):
    @api.doc('Return organization')
    def get(self, organization):
        '''Retrieves a Organization'''

        cursor = get_db().cursor(cursor_factory=RealDictCursor)

        cursor.execute("""SELECT *, 
        (select json_agg(row_to_json(oa)) from organization_affiliate oa where o.name LIKE oa.organization_name) as members,
        ARRAY(select name from model m where m.organization_name LIKE o.name) as models
        FROM organization o where o.name LIKE (%s)""",  (organization,))


        result = cursor.fetchone()
        result = dict(result)

        if 'username' in session:
            for member in result['members']:
                if member['username'] == session['username']:
                    result['role'] = member['affiliate_type']
                    break

        if 'role' not in result:
            result['role'] = 'guest'

        cursor.close()

        return jsonify(result)

    @api.doc('Create organization')
    def post(self, organization):
        '''Creates an Organization'''
        parser = reqparse.RequestParser()

        parser.add_argument('name', required=True, help="Organization name cannot be blank!")
        parser.add_argument('description', required=True, help="Description cannot be blank!")
        parser.add_argument('affiliation', required=True, help="Affiliation cannot be blank!")

        args = parser.parse_args()

        app.logger.info(args)

        cursor = get_db().cursor(cursor_factory=RealDictCursor)

        cursor.execute("INSERT INTO organization (name, description, affiliation) VALUES (%s, %s, %s);",  (args['name'], args['description'], args['affiliation']))

        cursor.execute("COMMIT;")

        cursor.close()

        return 200


@api.route('/<organization>/add/<username>')
@api.param('organization', 'The organization name')
@api.param('username', 'The username')
class OrganizationAdd(Resource):

    @api.doc('Create organization affiliate')
    def post(self, organization, username):
        if 'username' not in session:
            return "Invalid", 301

        cursor = get_db().cursor(cursor_factory=RealDictCursor)

        cursor.execute("SELECT username from organization_affiliate WHERE organization_name LIKE %s and affiliate_type LIKE 'owner'", (organization,))

        result = cursor.fetchone()

        if result is None:
            return "",404

        #insure this is the owner of the organization making the request
        if result['username'] != session['username']:
            return "", 301

        try:
            cursor.execute("INSERT INTO organization_affiliate VALUES (%s, %s, %s)" , (username, organization, 'member'))
        except UniqueViolation:
            return "User is already a member", 400
        cursor.execute("COMMIT")

        cursor.close()

        return "", 200



@api.route('/<organization>/invite/<username>')
@api.param('organization', 'The organization name')
@api.param('username', 'The username')
class OrganizationInvite(Resource):

    @api.doc('Create organization and username')
    def post(self, organization, username):
        if 'username' not in session:
            return "Invalid", 301

        cursor = get_db().cursor(cursor_factory=RealDictCursor)

        cursor.execute("SELECT username from organization_affiliate WHERE organization_name LIKE %s and affiliate_type LIKE 'owner'", (organization,))

        result = cursor.fetchone()

        if result is None:
            return "",404

        #insure this is the owner of the organization making the request
        if result['username'] != session['username']:
            return "", 301


        cursor.execute("SELECT from organization_affiliate WHERE organization_name LIKE %s and username LIKE %s", (organization, username))

        if cursor.fetchone() is not None:
            return "User already in organization", 400

        try:
            cursor.execute("INSERT INTO invitation VALUES (%s, %s, %s, %s)", (organization, username, result['username'], 'pending'))
        except UniqueViolation:
            return "Invitation already Exists", 400
        except ForeignKeyViolation:
            return "User does not exists", 400
        cursor.execute("COMMIT")

        cursor.close()












