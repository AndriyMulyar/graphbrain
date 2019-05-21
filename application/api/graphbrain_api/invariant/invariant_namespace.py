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


api = Namespace('invariant', description='Invariant related API calls')

@api.route('/')
class Invariant(Resource):

    @api.doc('Retrieves all graph invariants')
    @api.doc(responses={200: 'Invariants successfully retrieved'})
    def get(self):
        """
        Retrieves a invariant from the GraphBrain
        """
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("select invariant from public.invariants;")
                result = [row['invariant'] for row in cursor.fetchall()]

        return jsonify({'invariants': result})

@api.route('/<invariant>')
class Invariant(Resource):

    @api.doc('Retrieve a graph invariant')
    @api.doc(responses={404: 'Invariant not in database', 200: 'Invariant successfully retrieved'})
    def get(self, invariant):
        """
        Retrieves a invariant from the GraphBrain
        """
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("select * from public.invariants where invariant like %s;", (invariant,))
                result = cursor.fetchone()

        if result is None:
            return "Invariant not in database", 404

        return jsonify(result)





