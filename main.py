import base64
import os
import random
import re
import shutil
from contextlib import suppress

import aiohttp
import cv2
import pytesseract
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

ERROR = "error"
TASK_D = {
    "背包": "backpack",
    "狗": "dog",
    "火车": "train",
    "自行车": "bicycle",
    "汽车": "car",
    "猫": "cat",
    "沙发": "couch",
    "马": "horse",
    "船": "ship",
    "球": "sports ball",
}


class TaskImageRequest(BaseModel):
    task_image: str


@app.get("/health")
async def health():
    return {"status": "ok"}


def preprocess_image(image_path, thresh):
    image = cv2.imread(image_path)
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
    return ''.join(filter(lambda x: x not in {"一", "二", "三", "人"}, chinese_matches))


def to_normal_list(s):
    return " ".join((" ".join(s.split(',')).split("，"))).split()


def extract_task_metadata(image_path):
    custom_config = "--oem 3 --psm 6 --user-words whitelist.txt"
    raw_text = pytesseract.image_to_string(image_path, lang="chi_sim", config=custom_config)
    tokens = [token for token in to_normal_list(raw_text) if token]
    labels = [TASK_D[token] for token in tokens if token in TASK_D]
    return {
        "raw_text": raw_text.strip(),
        "tokens": tokens,
        "labels": labels,
        "matched": len(labels) > 0,
    }


async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                image_data = await response.read()
                name = generate_random_name() + ".jpg"
                with open(name, 'wb') as image_file:
                    image_file.write(image_data)
                return name
            return ERROR


@app.get('/chinese_symbols_image_analyse/process_image')
async def process_image(image_url: str):
    image_path = await download_image(image_url)

    if image_path == ERROR:
        raise HTTPException(status_code=400, detail="Image download failed")

    try:
        for thresh in (35, 235):
            result_path = preprocess_image(image_path, thresh)

            custom_config = "--oem 3 --psm 11"
            data = pytesseract.image_to_data(
                result_path,
                lang="chi_sim",
                config=custom_config,
                output_type=pytesseract.Output.DICT,
            )
            os.remove(result_path)

            count = 0
            for i in range(len(data['text'])):
                if not filter_chinese_text(data["text"][i]):
                    continue

                confidence = data['conf'][i]
                if confidence >= 90:
                    count += len(list("".join(data["text"][i].split())))

            if count >= 3:
                return False

        return True
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@app.post('/chinese_symbols_image_analyse/extract_task')
async def extract_task(request: TaskImageRequest):
    try:
        image_data = base64.b64decode(request.task_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 task image") from exc

    image_path = generate_random_name() + ".png"
    with open(image_path, 'wb') as image_file:
        image_file.write(image_data)

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Task image decode failed")

        resized_image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
        cv2.imwrite(image_path, resized_image)
        return extract_task_metadata(image_path)
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    with suppress(FileNotFoundError):
        shutil.rmtree("images")
    os.mkdir("images")
    uvicorn.run(app, host="0.0.0.0", port=5000)
