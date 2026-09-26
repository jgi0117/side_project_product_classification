from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from fastapi.testclient import TestClient
    from yolo_benchmark import rtmdet_api
except ModuleNotFoundError:
    TestClient = None

from yolo_benchmark.rtmdet_x import decide_top1


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        assert image.shape == (8, 8, 3)
        return decide_top1([(73, .9)])


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are not installed")
class RTMDetApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
        assert ok
        cls.image = encoded.tobytes()

    def test_health_and_single_label_prediction(self):
        predictor = FakePredictor()
        with patch.object(rtmdet_api, "RTMDetXPredictor", return_value=predictor), \
                patch.dict(os.environ, {"RTMDET_API_KEY": ""}):
            with TestClient(rtmdet_api.app) as client:
                self.assertEqual(client.get("/health").json(),
                                 {"status": "ok", "model": "rtmdet-x", "top_k": 1})
                response = client.post("/predict", files={"file": ("image.png", self.image, "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "book")
        self.assertEqual(response.json()["top_k"], 1)
        self.assertEqual(predictor.calls, 1)

    def test_bad_image_and_api_key(self):
        predictor = FakePredictor()
        with patch.object(rtmdet_api, "RTMDetXPredictor", return_value=predictor), \
                patch.dict(os.environ, {"RTMDET_API_KEY": "secret"}):
            with TestClient(rtmdet_api.app) as client:
                missing_key = client.post("/predict", files={"file": ("image.png", self.image, "image/png")})
                invalid = client.post("/predict", headers={"X-API-Key": "secret"},
                                      files={"file": ("bad.png", b"not an image", "image/png")})
        self.assertEqual(missing_key.status_code, 401)
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(predictor.calls, 0)


if __name__ == "__main__":
    unittest.main()
