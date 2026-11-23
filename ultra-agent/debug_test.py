from app import app

client = app.test_client()

resp = client.get("/api/users")
print("Status:", resp.status_code)
data = resp.get_json()
print("Body:", data)
print("Headers:", dict(resp.headers))