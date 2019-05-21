from flask import Blueprint
from flask_restplus import Api



blueprint = Blueprint('api', __name__, url_prefix='/api')

api = Api(blueprint,
          title='GraphBrain API',
          version='1.0',
          description='An API interfacing the GraphBrain database')

from .graph import graph_namespace
from .property import property_namespace
from .invariant import invariant_namespace

api.add_namespace(graph_namespace, path='/graph')
api.add_namespace(property_namespace, path='/property')
api.add_namespace(invariant_namespace, path='/invariant')


