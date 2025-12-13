import os
import sys
from pathlib import Path
import yaml

from verna.config import parse_config, Sections

TEMPLATE_PATH = Path('.do/app.template.yaml')
OUTPUT_PATH = Path('.do/app.yaml')


def upsert_env(envs: list[dict], key: str, value: str) -> None:
    for env in envs:
        if env.get('key') == key:
            env.update({'value': value, 'scope': 'RUN_TIME', 'type': 'SECRET'})
            return
    envs.append({'key': key, 'value': value, 'scope': 'RUN_TIME', 'type': 'SECRET'})


def main() -> int:
    cfg = parse_config(
        sections=[Sections.DB, Sections.TELEGRAM, Sections.AI],
        require_db=True,
        require_tg=True,
    )

    if not TEMPLATE_PATH.exists():
        sys.exit(f'Template not found: {TEMPLATE_PATH}')

    with TEMPLATE_PATH.open(encoding='utf-8') as f:
        data = yaml.safe_load(f)

    secrets = {
        'VERNA_API_KEY': cfg.api_key,
        'VERNA_API_BASE_URL': cfg.api_base_url,
        'VERNA_TG_BOT_TOKEN': cfg.tg_bot_token,
        'VERNA_TG_CHAT_ID': cfg.tg_chat_id,
        'VERNA_DB_CONN_STRING': cfg.db_conn_string,
    }

    jobs = data.get('jobs', [])
    if not jobs:
        sys.exit('No jobs found in template')

    for job in jobs:
        envs = job.setdefault('envs', [])
        if not isinstance(envs, list):
            sys.exit('Parse error: unexpected `envs` type')
        for key, value in secrets.items():
            upsert_env(envs, key, value)

    fd = os.open(OUTPUT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, explicit_start=True)

    print(f'Wrote {OUTPUT_PATH} (updated {len(jobs)} job(s)) with mode 600')
    return 0


if __name__ == '__main__':
    sys.exit(main())
