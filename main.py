import os
import random
import time

import cv2

from fastapi import FastAPI, HTTPException


import uvicorn
import re


import aiohttp
import pytesseract

app = FastAPI()

ERROR = "error"


def preprocess_image(image_path, thresh):
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    new_path = generate_random_name() + ".jpg"
    cv2.imwrite(new_path, binary_image)

    return new_path


def generate_random_name():
    return "images/" + ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(15))


def filter_chinese_text(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    chinese_matches = chinese_pattern.findall(text)
    return ''.join(filter(lambda x: x != "一" and x != "二" and x != "三" and x != "人", chinese_matches))


async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                image_data = await response.read()

                name = generate_random_name() + ".jpg"

                with open(name, 'wb') as image_file:
                    image_file.write(image_data)

                return name
            else:
                return ERROR


@app.get('/chinese_symbols_image_analyse/process_image')
async def process_image(image_url: str):
    image_path = await download_image(image_url)

    if image_path == ERROR:
        raise HTTPException(status_code=400, detail="Image download failed")

    for trash in (35, 235):
        print(f"Trash {trash}")
        result_path = preprocess_image(image_path, trash)

        custom_config = "--oem 3 --psm 11"
        data = pytesseract.image_to_data(result_path, lang="chi_sim", config=custom_config, output_type=pytesseract.Output.DICT)
        os.remove(result_path)
        # print(data)

        cnt = 0
        for i in range(len(data['text'])):
            # print(row)
            if not filter_chinese_text(data["text"][i]):
                continue

            confidence = data['conf'][i]
            print(confidence, data["text"][i])

            if confidence >= 90:
                cnt += len(list("".join(data["text"][i].split())))

        print(f"Cnt: {cnt}")
        if cnt >= 3:
            os.remove(image_path)
            return False

    os.remove(image_path)
    return True


if __name__ == "__main__":
    import shutil

    directory_path = "images"  # Replace with your directory path
    shutil.rmtree(directory_path)

    os.mkdir("images")
    uvicorn.run(app, host="0.0.0.0", port=5000)
