from flask_restplus import Namespace, Resource, reqparse
from psycopg2.extras import RealDictCursor
from psycopg2.errors import UniqueViolation
from flask import current_app as app, jsonify, session, g, request
import re, json
from ..data import get_db, get_conn
from sage.all import *
from sage.graphs.graph_input import from_graph6


api = Namespace('graph', description='Graph related API calls')


@api.route('/')
class Graph(Resource):

    @api.doc('Retrieves all graphs that have been computationally processed by the GraphBrain. These are ready to be conjectured on.')
    def get(self):
        """Retrieves all graphs that have properties and invariants computed"""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT graph6 from  public.graph where processed_status like 'processed';")
                result = [row['graph6'] for row in cursor.fetchall()]

        return jsonify(result)

    @api.doc(params={'graph6': 'The graph6 string of the graph you would like to retrieve.'})
    @api.doc(responses={400: 'Invalid graph6 string', 404: 'Graph not in database', 200: 'Graph successfully retrieved'})
    @api.doc('Return canonical graph6 string from database with any computed properties and invariants')
    def post(self):
        """
        Retrieves a graph from the GraphBrain with computed properties and invariants
        """

        parser = reqparse.RequestParser()
        parser.add_argument('graph6', required=True,
                            help="Must include if all currently graphs currently marked as processing should be set to queued")
        args = parser.parse_args()

        graph6 = args['graph6']

        #secretly accept int's representing graph id's local to the GraphBrain.
        if not graph6.isdigit():
            from sage.graphs.graph import Graph
            G = Graph()

            try:
                from_graph6(G, str(graph6))
            except RuntimeError as error:
                return str(error), 400

            canon_g6 = G.canonical_label(algorithm='sage').graph6_string()

        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if graph6.isdigit():
                    cursor.execute("select * from graph g where g.id = %s;", (graph6,))
                else:
                    cursor.execute("select * from graph g where g.graph6 = %s;", (canon_g6,))
                graph = cursor.fetchone()

                if not graph:
                    return "Graph not in database", 404
                graph = dict(graph)
                graph['properties'] = dict()
                graph['invariants'] = dict()

                #Retrieve properties
                cursor.execute("select p.property, pv.value from properties p join property_value pv on p.id = pv.property_id where pv.graph_id = %s;", (graph['id'],))
                properties = cursor.fetchall()
                if properties:
                    graph['properties'].update({row['property']: row['value'] for row in properties})

                #Retrieve invariants
                cursor.execute("select invariant, value from invariants i join invariant_value iv on i.id = iv.invariant_id where iv.graph_id = %s;", (graph['id'],))
                invariants = cursor.fetchall()
                if invariants:
                    graph['invariants'].update({row['invariant']: row['value'] for row in invariants})

        return jsonify(graph)

    @api.doc(params={'graph6': 'A graph6 string to add'})
    @api.doc('Add a graph to the database')
    @api.doc(responses={400: 'Invalid graph6 string',409: 'Graph already in database', 200: 'Graph successfully added'})
    def put(self):
        """
        Inserts a graph into the GraphBrain
        """
        parser = reqparse.RequestParser()
        parser.add_argument('graph6', required=True,
                            help="Must include if all currently graphs currently marked as processing should be set to queued")
        args = parser.parse_args()

        graph6 = args['graph6']

        from sage.graphs.graph import Graph
        G = Graph()

        try:
            from_graph6(G, str(graph6))
        except RuntimeError as error:
            return str(error), 400

        canon_g6 = G.canonical_label(algorithm='sage').graph6_string()

        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                try:
                    cursor.execute("INSERT INTO public.graph(graph6) VALUES (%s)", (canon_g6,))
                    conn.commit()
                except UniqueViolation:
                    return 'Graph already exists (canonical sage g6 string): %s' % canon_g6, 409

        return canon_g6, 201


#Insure that on start up all processing graphs are set to queued
@api.route('/poll')
class GraphComputationPoll(Resource):

    @api.doc('Polls a graph for computation')
    @api.doc(responses={404: 'No graphs in queue', 200: 'Graph successfully polled for processing'})
    def get(self):
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT id, graph6 from public.graph where processed_status like 'queued' ORDER BY LENGTH (graph6) LIMIT 1")
                result = cursor.fetchone()
                if result is None:
                    return "No graphs in queue", 404
                cursor.execute("UPDATE public.graph set processed_status = 'processing' where id = %s", (result['id'],))
                conn.commit()

        return jsonify(result)

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('reset_queue', required=True, help="Must include if all currently graphs currently marked as processing should be set to queued")
        args = parser.parse_args()

        if args['reset_queue']:
            with get_conn() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute("UPDATE public.graph set processed_status = 'queued' where processed_status LIKE 'processing';")
                        conn.commit()
            return "Queue Successfully Reset", 200

    def put(self):

        args = json.loads(request.data)

        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                try:
                    if args['properties'] or args['invariants']:
                        for property, value in args['properties']:
                            #app.logger.info("Inserting property: %s" % property)
                            try:
                                cursor.execute("INSERT INTO property_value(graph_id, property_id, value) VALUES (%s, (select id from properties where property like %s), %s)  ON CONFLICT (graph_id, property_id) DO UPDATE SET graph_id = excluded.property_id, property_id = excluded.property_id",
                                               (args['id'], property, value))
                            except UniqueViolation:
                                continue
                        for invariant, value in args['invariants']:
                            #app.logger.info("Inserting invariant: %s" % invariant)
                            try:
                                cursor.execute("INSERT INTO invariant_value(graph_id, invariant_id, value) VALUES (%s, (select id from invariants where invariant like %s), %s) ON CONFLICT (graph_id, invariant_id) DO UPDATE SET graph_id = excluded.graph_id, invariant_id = excluded.invariant_id",
                                               (args['id'], invariant, value))
                            except UniqueViolation:
                                continue
                    cursor.execute("UPDATE graph set processed_status='processed' where id = %s", (args['id'],))
                    conn.commit()
                except BaseException as error:
                    app.logger.info(error)
                    return str(error), 400
        return "Successfully added computation to database", 200








