import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baroreflex_pipeline.data.download import _download_with_python, _is_complete_file


class DownloadTests(unittest.TestCase):
    def test_wfdb_file_must_match_header_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "0001.hea").write_text(
                "0001 2 1000 3\n0001.dat 16 1/mV 16 0 0 0 0 ECG\n"
                "0001.dat 16 1/mmHg 16 0 0 0 0 NIBP\n",
                encoding="utf-8",
            )
            data = root / "0001.dat"
            data.write_bytes(b"\0" * 11)
            self.assertFalse(_is_complete_file(data))
            data.write_bytes(b"\0" * 12)
            self.assertTrue(_is_complete_file(data))

    def test_python_fallback_mirrors_nested_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "destination"
            listings = {
                "https://example.test/data/": ["../", "metadata.csv", "nested/"],
                "https://example.test/data/nested/": ["../", "0001.hea"],
            }

            def fake_download(url: str, output: Path) -> None:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(url.rsplit("/", 1)[-1], encoding="utf-8")

            with (
                patch(
                    "baroreflex_pipeline.data.download._directory_links",
                    side_effect=lambda url: listings[url],
                ),
                patch(
                    "baroreflex_pipeline.data.download._download_file", side_effect=fake_download
                ),
            ):
                _download_with_python("https://example.test/data/", destination)

            self.assertEqual((destination / "metadata.csv").read_text(), "metadata.csv")
            self.assertEqual((destination / "nested" / "0001.hea").read_text(), "0001.hea")


if __name__ == "__main__":
    unittest.main()
