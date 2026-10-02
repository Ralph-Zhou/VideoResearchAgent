"""Offline regressions for paper behavior and standalone release paths."""
import copy
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('scaffold', ['standard', 'hybrid'])
def test_agent_preserves_early_frames_and_answers_autonomously(monkeypatch, scaffold, tmp_path):
    from video_agent.config import AppConfig
    from video_agent.agent import video_research_agent as agent_module
    cfg = AppConfig()
    cfg.agent.scaffold_mode = scaffold
    cfg.agent.max_iterations = 8
    cfg.agent.full_trajectory = True
    cfg.agent.full_trajectory_dir = str(tmp_path)
    cfg.logging.save_trajectory = False
    cfg.logging.print_steps = False
    calls = []
    schema = {'type': 'function', 'function': {'name': 'watch_video', 'parameters': {'type': 'object', 'properties': {}}}}

    class FakeTools:
        def __init__(self, cfg):
            pass
        def get_openai_tools(self):
            return [schema]
        def execute(self, name, args):
            return SimpleNamespace(text='Observed scene', frames=[SimpleNamespace(timestamp=0.0, image_b64='ZmFrZQ==')])

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs['messages']))
        count = len(calls)
        tool_calls = [] if count == 6 else [SimpleNamespace(id=f'call{count}', type='function', function=SimpleNamespace(name='watch_video', arguments='{}'))]
        message = SimpleNamespace(role='assistant', content='<answer>blue</answer>' if count == 6 else None, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    monkeypatch.setattr(agent_module, 'ToolRegistry', FakeTools)
    monkeypatch.setattr(agent_module, 'OpenAI', lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    agent = agent_module.VideoResearchAgent(cfg)
    result = agent.run('Find the color')
    assert result['final_answer'] == 'blue'
    assert len((tmp_path / 'all.jsonl').read_text().splitlines()) == 1
    assert len(calls) == 6
    images = [item for message in calls[-1] if isinstance(message.get('content'), list)
              for item in message['content'] if item.get('type') == 'image_url']
    assert len(images) == 5  # Older than the former 3-turn pruning window.
    assert not hasattr(agent, '_prune_old_images')


@pytest.mark.parametrize('source', ['video_research_train', 'video_research_val_browsecomp'])
@pytest.mark.parametrize('solution,correct,expected', [
    ('<answer>blue</answer>', 1.0, 1.1),
    ('<answer>red</answer>', 0.0, 0.1),
    ('blue', 1.0, 0.0),
    ('<answer>red</answer><answer>blue</answer>', 1.0, 1.1),
])
def test_terminal_reward(monkeypatch, source, solution, correct, expected):
    reward = _load('release_reward', 'verl-video/verl/utils/reward_score/video_research_judge.py')
    monkeypatch.delenv('JUDGER_FORMAT_WEIGHT', raising=False)
    seen = []
    def judge(question, gold, candidate):
        seen.append(candidate)
        return correct, 'MATCH' if correct else 'MISMATCH'
    monkeypatch.setattr(reward, '_llm_judge', judge)
    result = reward.compute_score(source, solution, {'answer': 'blue'}, {})
    assert result['score'] == expected
    if '<answer>' in solution:
        assert seen == [reward._extract_answer(solution)]
    else:
        assert not seen


def test_rdr_never_invents_urls_and_keeps_input_unchanged():
    rdr = _load('release_rdr', 'verl-video/video_search_sim/video_search_sim/verl_tools/domain_randomization.py')
    candidates = [{'url': f'https://www.youtube.com/watch?v={i:011d}', 'title': f'Video {i}', 'snippet': 'A detailed scene'} for i in range(50)]
    before = copy.deepcopy(candidates)
    allowed = {x['url'] for x in candidates}
    for seed in range(30):
        result, _ = rdr.apply_domain_randomization(candidates, gold_urls={candidates[0]['url']}, config={'enable': True}, rng=random.Random(seed), display_topk=10)
        assert {x['url'] for x in result} <= allowed
    assert candidates == before


@pytest.mark.parametrize('rdr_enabled', [True, False])
def test_rl_data_conversion_preserves_questions_and_aligns_splits(tmp_path, monkeypatch, rdr_enabled):
    import json
    import pyarrow.parquet as pq
    converter = _load('release_data', 'task_generation/prepare_rl_data.py')
    train = tmp_path / 'tasks.jsonl'
    validation = tmp_path / 'benchmark.jsonl'
    gold_url = 'https://www.youtube.com/watch?v=example0001'
    train.write_text(json.dumps({
        'status': 'accept', 'seed': 'sample', 'question': 'Find the color', 'answer': 'blue',
        'graph': {'entities': [{'video': {'url': gold_url}}]},
    }) + '\n' + json.dumps({'status': 'reject', 'question': 'Excluded'}) + '\n')
    validation.write_text(json.dumps({'row_id': 'bc1', 'question': 'Find the person', 'answer': 'Example'}) + '\n')
    output = tmp_path / 'rl'
    args = ['prepare_rl_data.py', '--train_jsonl', str(train), '--val_browsecomp', str(validation), '--out_dir', str(output)]
    if not rdr_enabled:
        args.append('--disable_domain_randomization')
    monkeypatch.setattr(sys, 'argv', args)
    assert converter.main() == 0
    tr, val = [pq.read_table(output / f'{split}.parquet') for split in ('train', 'val')]
    assert tr.schema == val.schema
    assert tr.num_rows == val.num_rows == 1
    tr, val = tr.to_pylist()[0], val.to_pylist()[0]
    assert tr['prompt'][-1]['content'] == 'Find the color'
    assert val['prompt'][-1]['content'] == 'Find the person'
    assert gold_url not in json.dumps(tr['prompt'])
    assert tr['reward_model']['ground_truth']['gold_urls'] == [gold_url]
    train_kwargs = tr['extra_info']['tools_kwargs']['search_youtube']['create_kwargs']
    val_kwargs = val['extra_info']['tools_kwargs']['search_youtube']['create_kwargs']
    assert train_kwargs['backend'] == 'local'
    assert bool(train_kwargs.get('domain_randomization')) == rdr_enabled
    assert val_kwargs['backend'] == 'remote'
    assert not val_kwargs.get('domain_randomization')
