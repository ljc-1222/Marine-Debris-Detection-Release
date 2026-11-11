"""Allow reviewed public guides while keeping private assets out of the index."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryHygieneTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='.test-hygiene-')
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        (self.repo / 'scripts').mkdir()
        shutil.copyfile(ROOT / 'scripts/check_repo_hygiene.sh', self.repo / 'scripts/check_repo_hygiene.sh')
        self.git('init', '--quiet')
        self.git('add', '--', 'scripts/check_repo_hygiene.sh')

    def git(self, *arguments):
        return subprocess.run(['git', *arguments], cwd=self.repo, check=True, capture_output=True, text=True)

    def stage(self, filename, content):
        path = self.repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.git('add', '--', filename)

    def check_hygiene(self):
        return subprocess.run(['bash', 'scripts/check_repo_hygiene.sh'], cwd=self.repo, capture_output=True, text=True)

    def test_reviewed_public_guides_are_allowed(self):
        self.stage('docs/inference.md', '# Inference\n')
        self.stage('docs/training.md', '# Training\n')
        result = self.check_hygiene()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_private_docs_and_generated_assets_are_rejected(self):
        for filename in ('docs/prompts/agent.md', 'docs/donors.json', 'reports/private.md', 'checkpoints/model.pt'):
            with self.subTest(filename=filename):
                self.stage(filename, 'private fixture\n')
                result = self.check_hygiene()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(filename, result.stderr)
                self.git('rm', '--cached', '--', filename)
                (self.repo / filename).unlink()

    def test_public_guide_cannot_hide_a_credential(self):
        self.stage('docs/training.md', 'fake fixture credential: ' + 'hf_' + 'a' * 32 + '\n')
        result = self.check_hygiene()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('possible credential', result.stderr)


if __name__ == '__main__':
    unittest.main()
