"""Download the pinned public teaching release without redistributing the source."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

URL = "https://github.com/jaredsmurray/teaching-data/releases/download/v2026.09/lending_club.zip"
DIGEST = "aac82e6f808d8b994f2c68edf67d30b64f04b9ad2d04a0aeed55d2528831ef34"

if __name__ == "__main__":
    with urlopen(URL, timeout=90) as response:
        archive = response.read()
    if sha256(archive).hexdigest() != DIGEST:
        raise SystemExit("Unexpected SHA-256 for teaching release")
    target = Path("data/lending_club.zip")
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(archive)
    print(target)
