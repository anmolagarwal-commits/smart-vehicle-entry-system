import gdown

url = "https://drive.google.com/uc?id=1uYx7bY6Qk0n5u3h1dWz8zG5Qw3kP8r9X"
gdown.download(url, "plate_model.pt", quiet=False)