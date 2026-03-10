import os
from google import genai

def test_api():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY または GOOGLE_API_KEY が設定されていません。")
        print("   例: export GEMINI_API_KEY='your-api-key'")
        return
    client = genai.Client(api_key=api_key)

    print("APIの通信テストを開始します...")
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='こんにちは、これは通信テストです。一言だけ返事をお願いします。'
        )
        print("✅ 成功！返答:", response.text)
    except Exception as e:
        print("❌ エラー発生:", e)

if __name__ == "__main__":
    test_api()
