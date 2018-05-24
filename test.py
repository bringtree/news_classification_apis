# 接口测试
import requests
import unittest


class TestApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.login_url = 'http://0.0.0.0:5000/news_recognition'

    def test_recognition(self):
        """
        测试news_recognition接口
        """
        data = {
            'news_title': "空军发布多语种宣传片战神绕岛新航迹含闽南语等",
        }
        response = requests.post(self.login_url, data=data).json()
        assert response['code'] == '0'
        data = {
            'news_title': "食品中毒，死亡",
        }
        response = requests.post(self.login_url, data=data).json()
        assert response['code'] == '1'
        data = {
            'news_title': "",
        }
        response = requests.post(self.login_url, data=data).json()
        assert response['code'] == '-1'

