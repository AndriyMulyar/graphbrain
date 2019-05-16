import unittest, requests

class TestModelNameSpace(unittest.TestCase):

    def test_model_create(self):

        data = {
            'name': 'Fidels Model',
            'organization_name': 'Fidels Organization',
            'description': 'Test description',
            'implementation_language': 'Java',
            'implementation_framework': 'WEKA',
            'model_domain': 'NLP',
            'model_subdomain': 'Language Processing',
            'task': 'Parse stuff for Fidel'
        }

        r = requests.post("http://localhost/api/model/%s" % data['name'], data=data)

        print(str(r.text))

        self.assertEqual('foo'.upper(), 'FOO')