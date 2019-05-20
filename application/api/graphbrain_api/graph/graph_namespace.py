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


api = Namespace('graph', description='Graph related API calls')


@api.route('/<graph6>')
class Graph(Resource):

    @api.doc(responses={400: 'Invalid graph6 string', 404: 'Graph not in database', 200: 'Graph successfully retrieved'})
    @api.doc('Return canonical graph6 string from database with any computed properties and invariants')
    def get(self, graph6):
        """
        Retrieves a graph from the GraphBrain with computed properties and invariants
        """

        from sage.graphs.graph import Graph
        G = Graph()

        try:
            from_graph6(G, str(graph6))
        except RuntimeError as error:
            return str(error), 400

        canon_g6 = G.canonical_label(algorithm='sage').graph6_string()

        cursor = get_db().cursor(cursor_factory=RealDictCursor)
        cursor.execute("select * from graph g where g.graph6 like %s;", (canon_g6,))
        graph = cursor.fetchone()

        if not graph:
            return "Graph not in database", 404
        graph = dict(graph)
        graph['properties'] = dict()
        graph['invariants'] = dict()

        #Retrieve properties
        cursor.execute("select p.property, pv.value from properties p join property_value pv on p.id = pv.property_id where pv.graph_id = 25849;", (graph['id'],))
        properties = cursor.fetchall()
        if properties:
            graph['properties'].update({row['property']: row['value'] for row in properties})

        #Retrieve invariants
        cursor.execute("select invariant, value from invariants i join invariant_value iv on i.id = iv.invariant_id where iv.graph_id = %s;", (graph['id'],))
        invariants = cursor.fetchall()
        if invariants:
            graph['invariants'].update({row['invariant']: row['value'] for row in invariants})

        cursor.close()
        return jsonify(graph)

    @api.doc('Add a graph to the database')
    @api.doc(responses={400: 'Invalid graph6 string',409: 'Graph already in database', 200: 'Graph successfully added'})
    def put(self, graph6):
        """
        Inserts a graph into the GraphBrain
        """
        from sage.graphs.graph import Graph
        G = Graph()

        try:
            from_graph6(G, str(graph6))
        except RuntimeError as error:
            return str(error), 400

        canon_g6 = G.canonical_label(algorithm='sage').graph6_string()


        cursor = get_db().cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("INSERT INTO public.graph(graph6) VALUES (%s)", (canon_g6,))
        except UniqueViolation:
            return 'Graph already exists (canonical sage g6 string): %s' % canon_g6, 409
        result = cursor.fetchall()
        cursor.close()

        return jsonify(result), 201





