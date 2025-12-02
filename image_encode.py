import base64

with open("autoencoder.jpg", "rb") as f:
    data = base64.b64encode(f.read()).decode()

print(data) 