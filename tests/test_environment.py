"""Exercise setup decisions without downloading or creating model environments."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import environment, setup_codetr


class EnvironmentSetupTest(unittest.TestCase):
    def test_mmcv_build_path_and_worker_limit_use_the_active_environment(self):
        with patch.object(setup_codetr.sysconfig, "get_path", return_value="/active-venv/bin"):
            values = setup_codetr.build_environment({"PATH": "/system/bin", "TORCHINDUCTOR_COMPILE_THREADS": "2"}, Path("/cuda"), Path("/overlay"), 4)
        self.assertEqual(values["MAX_JOBS"], "4")
        self.assertEqual(values["PATH"].split(environment.os.pathsep), ["/cuda/bin", "/active-venv/bin", "/system/bin"])
        self.assertEqual(values["TORCHINDUCTOR_COMPILE_THREADS"], "2")
        for invalid in (0, 17):
            with self.assertRaisesRegex(ValueError, "1..16"):
                setup_codetr.build_environment({}, Path("/cuda"), Path("/overlay"), invalid)

    def test_mmcv_refuses_silent_non_ninja_fallback(self):
        with patch.object(setup_codetr.importlib.metadata, "version", return_value="1.13.2"), patch.object(setup_codetr.shutil, "which", return_value=None), patch.object(setup_codetr.subprocess, "run") as execute:
            with self.assertRaisesRegex(RuntimeError, "Ninja executable"):
                setup_codetr.require_ninja({"PATH": "/active-venv/bin"})
            execute.assert_not_called()
        with patch.object(setup_codetr.importlib.metadata, "version", return_value="1.13.2"), patch.object(setup_codetr.shutil, "which", return_value="/active-venv/bin/ninja"), patch.object(setup_codetr.subprocess, "run") as execute:
            setup_codetr.require_ninja({"PATH": "/active-venv/bin"})
        self.assertEqual(execute.call_args_list[0].args[0], ["/active-venv/bin/ninja", "--version"])
        self.assertIn("verify_ninja_availability", execute.call_args_list[1].args[0][-1])
    def test_caches_stay_under_requested_root_and_preserve_compile_threads(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory, patch.dict(environment.os.environ, {"TORCHINDUCTOR_COMPILE_THREADS": "2"}):
            root = Path(directory)
            values = environment.caches(root)
            self.assertEqual(values["TORCHINDUCTOR_COMPILE_THREADS"], "2")
            for variable in environment.CACHE_SUBDIRECTORIES:
                self.assertTrue(Path(values[variable]).is_relative_to(root))
                self.assertTrue(Path(values[variable]).is_dir())
            with patch.dict(environment.os.environ, {"TRITON_CACHE_DIR": str(root / "caller-triton")}):
                environment.configure_caches(root)
                self.assertEqual(environment.os.environ["TRITON_CACHE_DIR"], str(root / "caller-triton"))
                self.assertEqual(environment.os.environ["TORCHINDUCTOR_COMPILE_THREADS"], "2")
    def test_cuda_auto_uses_matched_toolkit_and_explicit_choice_is_preserved(self):
        with patch.object(environment, "cuda_toolkit", return_value=(Path("/cuda"), "12.6")):
            self.assertEqual(environment.cuda_wheel_variant("auto"), "cu126")
            self.assertEqual(environment.cuda_wheel_variant("cu130"), "cu130")
        with patch.object(environment, "cuda_toolkit", return_value=(Path("/cuda"), "12.7")):
            with self.assertRaisesRegex(RuntimeError, "No default wheel mapping"):
                environment.cuda_wheel_variant("auto")

    def test_codetr_stream_patch_is_exact_and_refuses_changed_source(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory)
            path = source / "mmcv/parallel/_functions.py"
            path.parent.mkdir(parents=True)
            (source / "setup.py").write_text("flags = ['-std=c++14']\n")
            path.write_text("streams = [_get_stream(device) for device in target_gpus]\n")
            setup_codetr.patch_source(source)
            self.assertIn("std=c++17", (source / "setup.py").read_text())
            self.assertIn("torch.device('cuda', device)", path.read_text())
            with self.assertRaisesRegex(ValueError, "exactly once"):
                setup_codetr.patch_source(source)

    def test_codetr_overlay_preserves_active_python_and_task_paths(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime = Path(directory)
            (runtime / "site-packages").mkdir()
            package = runtime / "mmcv/mmcv"
            package.mkdir(parents=True)
            (package / "__init__.py").touch()
            with patch.object(environment, "cuda_toolkit", return_value=(Path("/cuda-12.6"), "12.6")), patch.dict(environment.os.environ, {"PYTHONPATH": "/existing"}, clear=False):
                result = environment.codetr_environment(runtime)
            self.assertEqual(result["CUDA_HOME"], "/cuda-12.6")
            self.assertEqual(result["PYTHONPATH"].split(environment.os.pathsep)[:2], [str(runtime / "mmcv"), str(runtime / "site-packages")])
            self.assertTrue(result["PYTHONPATH"].endswith("/existing"))


if __name__ == "__main__":
    unittest.main()
