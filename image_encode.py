import base64

input_file = "autoencoder.png" # Sửa tên file image nếu cần
output_file = "autoencoder_base64.txt"

with open(input_file, "rb") as f: 
    data = base64.b64encode(f.read()).decode()

with open(output_file, "w") as out:
    out.write(data)