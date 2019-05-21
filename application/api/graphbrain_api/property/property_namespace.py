from flask_restplus import Namespace, Resource, reqparse
from psycopg2.extras import RealDictCursor
from psycopg2.errors import UniqueViolation
from flask import current_app as app, jsonify, session
import re
from datetime import datetime
from ..data import get_db, get_conn
import bcrypt
from sage.all import *
from sage.graphs.graph_input import from_graph6


api = Namespace('property', description='Property related API calls')

@api.route('/')
class Property(Resource):

    @api.doc('Retrieves all graph properties')
    @api.doc(responses={200: 'Properties successfully retrieved'})
    def get(self):
        """
        Retrieves a property from the GraphBrain
        """
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("select property from public.properties;")
                result = [row['property'] for row in cursor.fetchall()]

        return jsonify({'properties': result})

@api.route('/<property>')
class Property(Resource):

    @api.doc('Retrieve a graph property')
    @api.doc(responses={404: 'Property not in database', 200: 'Property successfully retrieved'})
    def get(self, property):
        """
        Retrieves a property from the GraphBrain
        """
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("select * from public.properties where property like %s;", (property,))
                result = cursor.fetchone()


        if result is None:
            return "Property not in database", 404

        return jsonify(result)

    @api.doc('Adds a property')
    def put(self, property):
        """
        Adds a property to the database
        """

        return "Not implemented, this should add a property and compute it for all graphs",404






