from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("mmcv"), "Run with scripts/codetr_python.py")
class CoDETRDependencyTest(unittest.TestCase):
    def test_config_dump_formats_and_reloads_without_building_a_model(self):
        from mmcv import Config

        values = dict(model=dict(type="CoDETR", classes=34), sizes=[(1024, 768), (1280, 960)])
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix=".test-codetr-config-") as directory:
            source = Path(directory) / "source.py"
            source.write_text("model = dict(type='CoDETR', classes=34)\nsizes = [(1024, 768), (1280, 960)]\n")
            config = Config.fromfile(str(source))
            self.assertIn("CoDETR", config.pretty_text)
            path = Path(directory) / "training.py"
            config.dump(path)
            self.assertEqual(Config.fromfile(str(path))._cfg_dict.to_dict(), values)


if __name__ == "__main__":
    unittest.main()
