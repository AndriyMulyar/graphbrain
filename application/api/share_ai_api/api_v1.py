from flask import Blueprint
from flask_restplus import Api



blueprint = Blueprint('api', __name__, url_prefix='/api')

api = Api(blueprint,
          title='Share.ai API',
          version='1.0',
          description='An API interfacing the Share.ai database')

from .user import user_namespace
from .model import model_namespace
from .organization import organization_namespace
from .logs import log_namespace

api.add_namespace(user_namespace, path='/user') #register /user path with it's functionality
api.add_namespace(model_namespace, path='/model')
api.add_namespace(organization_namespace, path='/organization')
api.add_namespace(log_namespace, path='/logs')



