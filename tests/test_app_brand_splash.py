"""The code-drawn Gold Quant mark and the animated loading splash.

Runs under the offscreen platform; asserts geometry and paint coverage rather
than glyph rendering (offscreen font fallback draws boxes for text).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from src.app.ui import brand, theme  # noqa: E402
from src.app.ui.splash import GoldSplash  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
theme.apply(_app, "dark")


def _coverage(pixmap, step=8):
    """Fraction of sampled pixels that are non-transparent."""

    img = pixmap.toImage()
    w, h = img.width(), img.height()
    total = hit = 0
    for x in range(0, w, step):
        for y in range(0, h, step):
            total += 1
            if img.pixelColor(x, y).alpha() > 20:
                hit += 1
    return hit / max(1, total)


class LogoTests(unittest.TestCase):
    def test_pixmap_sizes_and_content(self):
        for size in (16, 48, 256):
            px = brand.logo_pixmap(size)
            self.assertFalse(px.isNull())
            self.assertEqual(px.width(), size)
            self.assertGreater(_coverage(px, step=max(1, size // 16)), 0.1)

    def test_filled_variant_covers_the_coin_face(self):
        transparent = _coverage(brand.logo_pixmap(128), step=4)
        filled = _coverage(brand.logo_pixmap(128, filled=True), step=4)
        self.assertGreater(filled, transparent)  # the dark face is opaque
        self.assertGreater(filled, 0.6)

    def test_app_icon_carries_every_declared_size(self):
        icon = brand.app_icon()
        widths = sorted(s.width() for s in icon.availableSizes())
        self.assertEqual(widths, sorted(brand.ICON_SIZES))

    def test_write_ico_produces_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mark.ico"
            brand.write_ico(path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


class SplashTests(unittest.TestCase):
    def test_constructs_paints_and_updates_status(self):
        splash = GoldSplash()
        try:
            splash.set_status("Testing splash")
            splash.show()
            _app.processEvents()
            grab = splash.grab()
            self.assertFalse(grab.isNull())
            self.assertGreater(_coverage(grab, step=10), 0.5)
        finally:
            splash.close()

    def test_finish_into_fades_and_releases(self):
        splash = GoldSplash()
        window = QtWidgets.QWidget()
        try:
            splash.show()
            window.show()
            splash.finish_into(window)
            for _ in range(60):  # let the 260 ms fade animation run out
                _app.processEvents()
            self.assertIsNotNone(splash._fade)
        finally:
            splash.close()
            window.close()


if __name__ == "__main__":
    unittest.main()
