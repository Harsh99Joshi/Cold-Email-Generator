import re

import requests
from bs4 import BeautifulSoup


def clean_text(text):
    text = re.sub(r"<[^>]*?>", "", text)
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return " ".join(text.split()).strip()


def load_job_text(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ColdEmailGenerator/1.0)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ")
