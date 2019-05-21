import unittest, requests

class TestInvariantNameSpace(unittest.TestCase):

    def test_get_invariant(self):

        data = {
            'invariant': 'max_degree',
        }

        r = requests.get("http://localhost/api/invariant/"+data['invariant'], data=data)

        self.assertEqual(r.status_code, 200)

    def test_get_fake_invariant(self):

        data = {
            'invariant': 'test_invariant',
        }

        r = requests.get("http://localhost/api/invariant/"+data['invariant'], data=data)

        self.assertEqual(r.status_code, 404)




if __name__ == '__main__':
    unittest.main()