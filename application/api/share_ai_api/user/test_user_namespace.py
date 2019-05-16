import unittest, requests

class TestUserNameSpace(unittest.TestCase):

    def test_user_login(self):

        data = {
            'username': 'Andriy',
            'password': 'andriypassword'
        }

        session = requests.Session()


        r = session.post("http://localhost/api/user/%s/login" % data['username'], data=data)

        print(session.cookies.get_dict())

        r = session.post("http://localhost/api/user/%s/logout" % data['username'], data=data)

        print(session.cookies.get_dict())


        session.close()


        self.assertEqual(r.status_code, 200)