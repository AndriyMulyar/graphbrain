import unittest, requests

class TestGraphNameSpace(unittest.TestCase):

    def test_graph_add(self):

        data = {
            'graph6': 'Andriy'
        }

        r = requests.post("http://localhost/api/graph/", data=data)
        print(r.content)

        self.assertEqual(r.status_code, 200)



if __name__ == '__main__':
    unittest.main()