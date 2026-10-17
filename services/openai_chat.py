import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_answer_from_openai(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # hoặc gpt-3.5-turbo tuỳ bạn
            messages=[
                {"role": "system", "content": "Bạn là trợ lý luật"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"❌ Lỗi OpenAI: {str(e)}"
