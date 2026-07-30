import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.llm import client
response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[{"role":"user", "content":"say hi"}],
)
print(response.choices[0].message.content)