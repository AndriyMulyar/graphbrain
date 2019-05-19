from sage.all import *
from flask_restplus import Namespace, Resource, reqparse
from psycopg2.extras import RealDictCursor
from flask import current_app as app, jsonify, session
import re
from datetime import datetime
from ..data import get_db
import bcrypt
api = Namespace('graph', description='Graph related API calls')


@api.route('/')
class Graph(Resource):

    @api.doc('Return status and role of user making API call')
    def get(self):

        # cursor = get_db().cursor(cursor_factory=RealDictCursor)
        # cursor.execute("select property, count(graph) from public.prop_values GROUP BY property ORDER BY count(graph);")
        # result = cursor.fetchall()
        # cursor.close()
        g = graphs.PetersenGraph()
        return jsonify({'graph': g.graph6_string()})

    @api.doc('Add a graph to the database')
    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument('graph6', required=True, help="Missing graph6 string")


        cursor = get_db().cursor(cursor_factory=RealDictCursor)
        cursor.execute("select property, count(graph) from public.prop_values GROUP BY property ORDER BY count(graph);")
        result = cursor.fetchall()
        cursor.close()

        return jsonify(result)


@api.route('/add/<graph>')
class AddGraph(Resource):

    @api.doc('Add a graph to the database')
    def get(self, graph):

        cursor = get_db().cursor(cursor_factory=RealDictCursor)
        cursor.execute("select property, count(graph) from public.prop_values GROUP BY property;")
        result = cursor.fetchall()
        cursor.close()

        return jsonify(result)
#
#
# @api.route('/<username>')
# @api.param('username', 'The username')
# class User(Resource):
#     @api.doc('Return user')
#     def get(self, username):
#         '''Retrieves a User'''
#
#         if 'username' in session and session['username'] == 'root' and username == 'root':
#             return jsonify({'role': 'root'})
#
#         cursor = get_db().cursor(cursor_factory=RealDictCursor)
#         cursor.execute("SELECT *, date_part('year', account.creation_time_stamp) FROM account JOIN user_account on \
#                        account.username = user_account.username\
#                        WHERE user_account.username = %s", (username, ))
#         result = cursor.fetchone()
#
#         cursor.execute("""
#                     select json_agg(row_to_json(oa)) AS organizations from organization_affiliate oa where username = %s """
#                        , (username, ))
#
#         organizations = cursor.fetchone()
#
#         cursor.execute("""
#                 SELECT json_agg(row_to_json(i)) AS invitations FROM invitation i WHERE recipient_username = %s """
#                        , (username, ))
#
#         invitations = cursor.fetchone()
#
#         cursor.execute("SELECT ARRAY_AGG(model_name) as owned_models FROM transactions WHERE username = %s", (username,))
#
#         models = cursor.fetchone()
#
#         if result is None:
#             return "",404
#
#         result = dict(result)
#         result.update(dict(organizations))
#         result.update(dict(invitations))
#
#         result.update(dict(models))
#
#         #handle session
#         if 'username' in session and session['username'] == username:
#             result['role'] = 'account_owner'
#
#         else:
#             result['role'] = 'guest'
#
#         result.pop('password')
#         # cursor = get_db().cursor(cursor_factory=RealDictCursor)
#         # cursor.execute("SELECT * FROM user_account WHERE username = '%s' " % username)
#         # result = cursor.fetchone()
#         cursor.close()
#
#         return jsonify(result)
#
#     @api.doc('Create user')
#     def post(self, username):
#         '''Create a User'''
#
#         parser = reqparse.RequestParser()
#
#         parser.add_argument('username', required=True, help="Username cannot be blank!")
#         parser.add_argument('password', required=True, help="Password cannot be blank!")
#         # parser.add_argument('role', required=True, help="Role cannot be blank!")
#         parser.add_argument('email', required=True, help="Email cannot be blank!")
#         parser.add_argument('first_name', required=True, help="First name cannot be blank!")
#         parser.add_argument('last_name', required=True, help="Last name cannot be blank!")
#         parser.add_argument('country', required=True, help="Country cannot be blank!")
#         parser.add_argument('state_province', required=True, help="State/province cannot be blank!")
#         parser.add_argument('city', required=True, help="City cannot be blank!")
#         parser.add_argument('address', required=True, help="Address cannot be blank!")
#         args = parser.parse_args()
#
#         app.logger.info(args)
#
#         password = args['password'].encode('utf8')
#
#         hashed = bcrypt.hashpw(password, bcrypt.gensalt())
#
#         insertPassword = hashed.decode('utf8')
#
#         cursor = get_db().cursor(cursor_factory=RealDictCursor)
#
#
#         insertQuery2 = "INSERT INTO account (username, password, creation_time_stamp, role)\
#                        VALUES (%s, %s, CURRENT_TIMESTAMP, %s);"
#
#         cursor.execute(insertQuery2,  (args['username'], insertPassword, "user"))
#
#         insertQuery = "INSERT INTO user_account (username, first_name, last_name, country, state_or_province,\
#                        city, street_address, email) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);" \
#
#
#         cursor.execute(insertQuery, (args['username'], args['first_name'], args['last_name'], args['country'],
#                                     args['state_province'],
#                                     args['city'], args['address'], args['email']))
#
#         cursor.execute("COMMIT;")
#
#         session['username'] = args['username']
#
#         cursor.close()
#
#         return 200
#
#
#     @api.doc('Update user')
#     def put(self, username):
#         '''Update a User'''
#
#         parser = reqparse.RequestParser()
#
#         parser.add_argument('first_name', required=True, help="First name cannot be blank!")
#         parser.add_argument('last_name', required=True, help="Last name cannot be blank!")
#         parser.add_argument('country', required=True, help="Country cannot be blank!")
#         parser.add_argument('state_province', required=True, help="State/province cannot be blank!")
#         parser.add_argument('city', required=True, help="City cannot be blank!")
#         parser.add_argument('address', required=True, help="Address cannot be blank!")
#         args = parser.parse_args()
#
#         app.logger.info(args)
#
#         cursor = get_db().cursor(cursor_factory=RealDictCursor)
#
#         insertQuery = "UPDATE user_account SET first_name=%s, last_name=%s, country=%s, state_or_province=%s,\
#                         city=%s, street_address=%s WHERE username = %s;"
#
#         cursor.execute(insertQuery, (args['first_name'], args['last_name'], args['country'], args['state_province'], args['city'], args['address'], username))
#
#         cursor.execute("COMMIT;")
#
#         cursor.close()
#
#         return "", 200
#
#
#
#
#
# @api.route('/<username>/invitation/respond')
# @api.param('username', 'The username')
# class UserInvitation(Resource):
#
#     @api.doc('responds to users invitation request')
#     def post(self, username):
#         parser = reqparse.RequestParser()
#         parser.add_argument('organization', required=True, help="Organization cannot be blank!")
#         parser.add_argument('response', required=True, help="response cannot be blank!")
#         args = parser.parse_args()
#
#         cursor = get_db().cursor(cursor_factory=RealDictCursor)
#
#         if args['response'] == 'accept':
#             cursor.execute("INSERT INTO organization_affiliate VALUES (%s, %s, %s)", (username, args['organization'],
#                                                                                      'member'))
#
#         cursor.execute("DELETE FROM invitation WHERE organization = %s AND recipient_username = %s",
#                        (args['organization'], username))
#
#         cursor.execute("COMMIT")
#
#         return "",201
#
#
#
# @api.route('/<username>/login')
# @api.param('username', 'The username')
# class UserLogin(Resource):
#
#     @api.doc('Logs in user')
#     def post(self, username):
#         parser = reqparse.RequestParser()
#         parser.add_argument('username', required=True, help="Username cannot be blank!")
#         parser.add_argument('password', required=True, help="Password cannot be blank!")
#         args = parser.parse_args()
#
#         cursor = get_db().cursor(cursor_factory=RealDictCursor)
#
#         cursor.execute("SELECT username, password FROM account WHERE username = %s", (args['username'], ))
#         result = cursor.fetchone()
#         cursor.close()
#
#         if bcrypt.checkpw(args['password'].encode('utf8'), result['password'].encode('utf8')):
#             session['username'] = result['username']
#             return "", 200
#
#         return "", 301
#
# @api.route('/<username>/logout')
# @api.param('username', 'The username')
# class UserLogout(Resource):
#
#     @api.doc('Logs a user out')
#     def post(self, username):
#
#         if 'username' in session and session['username'] == username:
#             session.pop('username')
#             return "",200
#
#         return "",406




