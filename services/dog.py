import requests
from typing import Optional, List

DOG_RANDOM = "https://dog.ceo/api/breeds/image/random"
DOG_RANDOM_N = "https://dog.ceo/api/breeds/image/random/{n}"


def fetch_random_dog_image(timeout: int = 8) -> Optional[str]:
    r = requests.get(DOG_RANDOM, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        return None
    return data.get("message")


def fetch_random_dog_images(n: int = 1, timeout: int = 8) -> List[str]:
    n = max(1, min(int(n), 50))
    url = DOG_RANDOM_N.format(n=n)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        return []
    msg = data.get("message")
    if isinstance(msg, list):
        return msg
    if isinstance(msg, str):
        return [msg]
    return []
