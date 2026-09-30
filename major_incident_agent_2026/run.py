import asyncio, json
from langgraph_sdk import get_client

async def main():
    client = get_client(url="http://127.0.0.1:2024")
    thread = await client.threads.create()
    payload = json.load(open("test_input.json", encoding="utf-8"))
    await client.runs.create(thread["thread_id"], "mia", input=payload)
    print("thread_id:", thread["thread_id"])

asyncio.run(main())