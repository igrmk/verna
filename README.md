Verna
=====

This tool translates words and texts between English and Russian,
saves advanced English words and phrases as cards for later memorisation,
and sends your daily cards with AI-generated example sentences to Telegram.

In many languages, the word for "translation" contains "ver".
This useless fact inspired the tool's name.
Also I like the film Miller's Crossing.

Prerequisites
-------------

1. OpenAI/OpenRouter/... API key (`--api-key`)
2. PostgreSQL database (`--db-conn-string`)
3. Your own Telegram bot (`--tg-bot-token`, `--tg-chat-id`)

Installation
------------

From the project directory run:

    pipx install . --force

Deploy
------

The daily card sender runs on DigitalOcean's App Platform. To create an app, run:

    python -m verna.create_spec
    doctl apps create --spec .do/app.yaml

Once it is deployed, you can update it with:

    python -m verna.create_spec
    doctl apps update <app id> --spec .do/app.yaml

Note that in both cases, you'll need to add the secrets manually in the UI afterwards.
