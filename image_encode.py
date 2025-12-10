import base64

file_prefix = "autoencoder" # Sửa tên file image nếu cần

with open(f"{file_prefix}.jpg", "rb") as f: 
    data = base64.b64encode(f.read()).decode()

with open(f"{file_prefix}_base64.txt", "w") as out:
    out.write(data)