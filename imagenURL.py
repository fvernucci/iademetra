import urllib.request
from PIL import Image
from io import BytesIO

def convert_url_to_image(url, save_path):
    try:
        with urllib.request.urlopen(url) as response:
            image_data = response.read()
            image = Image.open(BytesIO(image_data))

            image.save(save_path)
            print(f"Image saved to {save_path}")
    except urllib.error.URLError as e:
        print("Error fetching the URL:", e)
    except Exception as e:
        print("Error:", e)

# URL of the image you want to convert
image_url = "http://127.0.0.1:5000/get?image_url=https://res.cloudinary.com/dhnzx75kb/image/upload/v1692581180/original_15_nrnawm.jpg"

# Path where you want to save the downloaded image
save_path = "downloaded_image.jpg"

convert_url_to_image(image_url, save_path)
