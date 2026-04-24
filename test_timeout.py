import litellm
import httpx
import os

os.environ["OLLAMA_API_BASE"] = "http://127.0.0.1:11434"

litellm.drop_params = True
my_client = httpx.Client(timeout=3) # Small timeout to force error
try:
    response = litellm.completion(
        model="ollama/gemma4:26b",
        messages=[{"role": "user", "content": "hello"}],
        timeout=3, # Test if this works
        client=my_client
    )
    print("Success")
except Exception as e:
    print(f"Failed with: {type(e).__name__} - {e}")
