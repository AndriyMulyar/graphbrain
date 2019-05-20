import unittest, requests

class TestPropertyNameSpace(unittest.TestCase):

    def test_get_property(self):

        data = {
            'property': 'test_property',
        }

        r = requests.put("http://localhost/api/property/"+data['property'], data=data)

        self.assertEqual(r.status_code, 400)

    def test_get_fake_property(self):

        data = {
            'property': 'test_property',
        }

        r = requests.put("http://localhost/api/property/"+data['property'], data=data)

        self.assertEqual(r.status_code, 400)




if __name__ == '__main__':
    unittest.main()