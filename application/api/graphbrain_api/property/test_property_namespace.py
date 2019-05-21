import unittest, requests

class TestPropertyNameSpace(unittest.TestCase):

    def test_get_property(self):

        data = {
            'property': 'has_c4',
        }

        r = requests.get("http://localhost/api/property/"+data['property'])

        self.assertEqual(r.status_code, 200)

    def test_get_not_property(self):

        data = {
            'property': 'empty',
        }

        r = requests.get("http://localhost/api/property/"+data['property'])

        self.assertEqual(r.status_code, 404)




if __name__ == '__main__':
    unittest.main()