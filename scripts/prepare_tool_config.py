"""Resolve training-tool paths for a shared-filesystem Ray cluster."""
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', required=True)
    parser.add_argument('--service-url', default='http://127.0.0.1:8000')
    parser.add_argument('--execution-backend', choices=['local', 'remote_server'], default='local')
    parser.add_argument('--output', default='data/runtime/tool_config.yaml')
    args = parser.parse_args()
    corpus = Path(args.corpus_dir).expanduser().resolve()
    for name in ['url_to_path.json', 'videos.parquet']:
        if not (corpus / name).is_file():
            parser.error(f'Missing corpus artifact: {corpus / name}')
    config = yaml.safe_load((ROOT / 'verl-video/video_search_sim/configs/tool_config.yaml').read_text())
    for tool in config['tools']:
        cfg = tool['config']
        for key in ['corpus_dir', 'downloader_cache_dir', 'transcript_cache_dir']:
            if key in cfg:
                cfg[key] = str(corpus if key == 'corpus_dir' else (ROOT / cfg[key]).resolve())
        if 'retrieval_service_url' in cfg:
            cfg['retrieval_service_url'] = args.service_url.rstrip('/') + '/video_search'
        if 'watch_service_url' in cfg:
            cfg['watch_service_url'] = args.service_url.rstrip('/') + '/watch_video'
            cfg['execution_backend'] = args.execution_backend
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(output)

if __name__ == '__main__':
    main()
