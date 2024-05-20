import unittest
from unittest.mock import patch
from app import *

class TestApp(unittest.TestCase):

    def test_format_docs(self):
        docs = [Document(page_content="Page 1"), Document(page_content="Page 2"), Document(page_content="Page 3")]
        expected_result = "Page 1\n\nPage 2\n\nPage 3"
        self.assertEqual(format_docs(docs), expected_result)

    @patch("langchain.hub.pull")
    def test_prompt_template_from_template(self, mock_pull):
        mock_pull.return_value = prompt
        template = "This is a template"
        expected_result = PromptTemplate.from_template(template)
        self.assertEqual(prompt_template_from_template(template), expected_result)

    def test_all_questions(self):
        expected_questions = [
            "Which state, area, or territory is the reference law regarding to? The answer should be only the name of the state, area, or territory, without any explanation.",
            "What is the legal commitment name? It is the varbatim name of the commitment in law. Answer should be just the name (e.g. Renewable Portfolio Standard), without any explanation.",
            "demand reduction", "energy efficiency", "chp", "res chp", "msw", "anaerobic digestion", "coal mine methane", "tire derived", "advanced coal", "coal mine waste", "coal gasification", "natural gas", "dist generation", "recycled energy", "carbon free", "non ff nuclear carbon free", "partial carbon capture", "industrial cogeneration", "near zero carbon", and "new technology". "The answer should only include the name of the energy sources (the name from the text, do not change) and the option. Energy sources should be separated by bracket and newline. The energy name and its option should be separated by a colon. Do not provide any explanation.",
            "List energy sources that meet the criteria of RPS. Categorize the energy sources you found into the following options: \"biogas\", \"biomass\", \"liquid biofuel\", \"res fuel cells\", \"fuel cells\", \"hydroelectric\", \"small falling water\", \"small hydro\", \"solar\", \"solar small\", \"ocean tidal wave thermal\", \"wind\", \"wind small\", \"geothermal geoelectric\", \"nuclear\", \"landfill gas\", \"demand reduction\", \"energy efficiency\", \"chp\", \"res chp\", \"msw\", \"anaerobic digestion\", \"coal mine methane\", \"tire derived\", \"advanced coal\", \"coal mine waste\", \"coal gasification\", \"natural gas\", \"dist generation\", \"recycled energy\", \"carbon free\", \"non ff nuclear carbon free\", \"partial carbon capture\", \"industrial cogeneration\", \"near zero carbon\", and \"new technology\". The answer should only include the name of the energy sources (the name from the text, do not change) and the option. Energy sources should be separated by bracket and newline. The energy name and its option should be separated by a colon. Do not provide any explanation.",
            "List energy sources that receives multifold credit (e.g. doubled, tripled) towards RPS/CES commitment if any. The answer should only include the name of the energy sources and their corresponding multiple in numerical value(e.g. double = 2). Energy sources should be separated by bracket and newline. In each energy source, the energy name and its corresponding multiple should be separated by a colon. If no energy sources receive multifold credits, give \"None\" as the response. Do not provide any explanation.",
            "What is the date at which the commitment passed into law. It is ususally the date that the document is filed or approved. Answer the question in the data format YYYY/MM/DD, without any explanation.",
            "What's the RPS/CES commitment in the policy and what's the year by which the commitment must be met? If the policy has multiple commitments at different dates, show the commitment associated with the largest date. The answer should only include the percentage (e.g. 10%) and year, separated by a colon. Do not provide with any explanation.",
            "What is the RPS commitment for the year the policy was introduced? The answer should be a percentage (e.g. 10%). Do not answer with any explanation.",
            "What's the percentage of the RPS/CES commitment that is voluntary? If the policy has multiple commitments at different dates, show the voluntary percentage of the commitment associated with the largest date. The answer should be a percentage with two decimal places (e.g. 10.00%). If the text does not mention voluntary percentage, the answer should be 0.00%. Do not answer with any explanation."
        ]
        self.assertEqual(all_questions, expected_questions)

if __name__ == '__main__':
    unittest.main()