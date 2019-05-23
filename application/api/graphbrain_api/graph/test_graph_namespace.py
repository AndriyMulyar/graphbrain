import unittest, requests

class TestGraphNameSpace(unittest.TestCase):

    def test_graph_add_invalid(self):

        data = {
            'graph6': 'Andriy'
        }

        r = requests.put("http://localhost/api/graph/", data=data)

        self.assertEqual(r.status_code, 400)

    def test_graph_add_already_exists(self):

        data = {
            'graph6': 'C~'
        }

        r = requests.put("http://localhost/api/graph/" , data=data)

        self.assertEqual(r.status_code, 409)



    def test_success_retrieve_graph(self):
        data = {
            'graph6': 'C~'
        }

        r = requests.post("http://localhost/api/graph/", data=data)
        self.assertEqual(r.status_code, 200)

    def test_failure_retrieve_graph(self):
        data = {
            'graph6': 'not a graph'
        }

        r = requests.post("http://localhost/api/graph/", data=data)
        self.assertEqual(r.status_code, 200)



if __name__ == '__main__':
    unittest.main()