"""Exercise real Git checkout conversion without touching the project's index."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest
from app.core.config import REPOSITORY_ROOT


@pytest.mark.parametrize('use_policy', [False, True])
@pytest.mark.parametrize('input_crlf', [False, True])
def test_windows_checkout_preserves_audited_sources(tmp_path, use_policy, input_crlf):
    manifest = json.loads((REPOSITORY_ROOT /
        'backend/app/components/component3/experimental/v3_manifest.json').read_text())
    repo = tmp_path / 'staging'
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE)

    git('init', '--quiet')
    git('config', 'core.autocrlf', 'true')
    git('config', 'core.safecrlf', 'false')
    git('config', 'core.attributesFile', str(tmp_path / 'no-global-attributes'))
    if use_policy:
        (repo / '.gitattributes').write_bytes((REPOSITORY_ROOT / '.gitattributes').read_bytes())
    for relative, digest in manifest['source_sha256'].items():
        original = (REPOSITORY_ROOT / relative).read_bytes()
        assert sha256(original).hexdigest() == digest
        assert b'\r' not in original
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.replace(b'\n', b'\r\n') if input_crlf else original)
    # The scoped rules must not change unrelated Windows checkout behavior.
    (repo / 'unrelated.txt').write_bytes(b'ordinary text\n')
    git('add', '--all')  # Temporary Git index only; no commit is created.
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    git('checkout-index', '--all', '--prefix=' + checkout.as_posix() + '/')
    assert (checkout / 'unrelated.txt').read_bytes() == b'ordinary text\r\n'
    for relative, digest in manifest['source_sha256'].items():
        assert sha256(git('show', ':' + relative)).hexdigest() == digest
        actual = (checkout / relative).read_bytes()
        assert (sha256(actual).hexdigest() == digest) == use_policy
        assert (b'\r\n' in actual) != use_policy
