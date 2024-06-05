import PIL.Image
import google.generativeai as genai

GEMINI_API = 'AIzaSyDBMZ3et4NDnkl9Q-p71bkf1kkdlvAe49E'


class Gemini(object):
    def __init__(self, api_key=GEMINI_API):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro-vision')

    def generate(self, img, text):

        response = self.model.generate_content([text, img])
        # response.resolve()

        return response.text
