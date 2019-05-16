from flask import Blueprint
from flask_restplus import Api



blueprint = Blueprint('api', __name__, url_prefix='/api')

api = Api(blueprint,
          title='GraphBrain API',
          version='1.0',
          description='An API interfacing the GraphBrain database')

from .graph import graph_namespace

api.add_namespace(graph_namespace, path='/graph') #register /user path with it's functionality



