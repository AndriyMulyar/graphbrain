from flask_restplus import Namespace, Resource, reqparse
from psycopg2.extras import RealDictCursor
from psycopg2.errors import UniqueViolation
from flask import current_app as app, jsonify, session
import re
from datetime import datetime
from ..data import get_db
import bcrypt
from sage.all import *
from sage.graphs.graph_input import from_graph6


api = Namespace('property', description='Property related API calls')


@api.route('/<property>')
class Property(Resource):

    @api.doc('Retrieve a graph property')
    @api.doc(responses={404: 'Property not in database', 200: 'Property successfully retrieved'})
    def get(self):
        """
        Retrieves a property from the GraphBrain
        """
        cursor = get_db().cursor(cursor_factory=RealDictCursor)
        cursor.execute("select * from public.properties where property like %s;", (property,))
        result = cursor.fetchall()
        cursor.close()
        if not result:
            return "Property not in database", 404

        return jsonify(result), 200

    # @api.doc('Adds a new graph property')
    # def put(self):
    #     """
    #     Adds a property to the GraphBrain
    #     :return:
    #     """
    #     pass




