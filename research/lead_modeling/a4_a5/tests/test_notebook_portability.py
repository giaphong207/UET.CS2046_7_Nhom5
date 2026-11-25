"""Execute only notebook path discovery in an independent clean candidate copy."""
import json
from pathlib import Path
import shutil

import pytest

from diabetes_research.paths import COMPONENT_ROOT, REPOSITORY_ROOT

NOTEBOOKS = ('A4_Training_Protocol_v2.ipynb', 'A5_Evaluation_XAI_v2.ipynb')


@pytest.fixture
def clean_candidate(tmp_path):
    candidate = tmp_path / 'reviewed-repository'
    # Copy repository assets only, without Git, data, environments or run outputs.
    for directory in ('src/diabetes_research', 'shared_splits',
                      'research/lead_modeling/a4_a5/config',
                      'research/lead_modeling/a4_a5/scripts',
                      'research/lead_modeling/a4_a5/notebooks'):
        for source in (REPOSITORY_ROOT / directory).iterdir():
            if source.is_file() and source.suffix in ('.py', '.npz', '.json', '.yaml', '.ipynb'):
                dest = candidate / source.relative_to(REPOSITORY_ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
    for source in [REPOSITORY_ROOT / 'pyproject.toml', *REPOSITORY_ROOT.glob('requirements*.txt'),
                   *COMPONENT_ROOT.glob('requirements*.txt')]:
        dest = candidate / source.relative_to(REPOSITORY_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return candidate


@pytest.mark.parametrize('name', NOTEBOOKS)
@pytest.mark.parametrize('location', ['', 'research/lead_modeling/a4_a5',
                                      'research/lead_modeling/a4_a5/notebooks'])
def test_notebook_discovery_from_full_clean_copy(clean_candidate, monkeypatch, name, location):
    path = clean_candidate / 'research/lead_modeling/a4_a5/notebooks' / name
    notebook = json.loads(path.read_text())
    code = next(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    monkeypatch.chdir(clean_candidate / location)
    scope = {}
    exec(compile(code.split('DATA')[0], str(path), 'exec'), scope)
    assert scope['REPO'] == clean_candidate
    assert scope['ROOT'] == clean_candidate / 'research/lead_modeling/a4_a5'
    assert not (clean_candidate / '.git').exists()
    assert not (scope['ROOT'] / 'data').exists()


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_notebook_discovery_rejects_component_only_mount(clean_candidate, monkeypatch, name):
    component = clean_candidate / 'research/lead_modeling/a4_a5'
    notebook = json.loads((component / 'notebooks' / name).read_text())
    code = next(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    shutil.rmtree(clean_candidate / 'src')
    monkeypatch.chdir(component / 'notebooks')
    with pytest.raises(FileNotFoundError, match='full V3 repository copy'):
        exec(compile(code.split('DATA')[0], name, 'exec'), {})
