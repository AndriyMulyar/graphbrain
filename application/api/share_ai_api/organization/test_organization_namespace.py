import unittest, requests

class TestOrganizationNameSpace(unittest.TestCase):

    def test_organization_create(self):

        data = {
            'name': 'OpenAI - NLP',
            'user': 'Andriy'
        }

        r = requests.post("http://localhost/api/organization/%s/add/%s" % (data['name'], data['user']), data=data)

        print(str(r.text))

        self.assertEqual('foo'.upper(), 'FOO')