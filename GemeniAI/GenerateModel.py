import PIL.Image
import google.generativeai as genai

GEMINI_API = 'AIzaSyDBMZ3et4NDnkl9Q-p71bkf1kkdlvAe49E'


class Gemini(object):
    def __init__(self, api_key=GEMINI_API):
        """
        :param api_key: Gemini API key

        初始化GeminiAI
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro-vision')

    def generate(self, img, text):
        """
        :param img: 图片
        :param text: 文本
        :return: 生成的文本

        生成文本
        """
        response = self.model.generate_content([text, img])
        # response.resolve()

        return response.text
