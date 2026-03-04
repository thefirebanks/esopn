import time

from PIL import Image

from esopn.capture import Screenshot


def test_to_base64_does_not_mutate_original_image() -> None:
    image = Image.new("RGB", (2000, 1200), color=(255, 0, 0))
    shot = Screenshot(image=image, width=2000, height=1200, monitor=1, timestamp=time.time())

    before_size = shot.image.size
    _ = shot.to_base64(max_size=(400, 300))

    assert shot.image.size == before_size
