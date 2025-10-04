Verna
=====

This tool translates words and texts between English and Russian,
lets you save advanced English words and phrases as cards for later memorisation,
and sends your daily cards and a story using them in Telegram.

In many languages, the word for "translation" contains "ver", which inspired the tool's name.
Also I like the film Miller's Crossing.

Installation
------------

From the project directory run:

    pipx install . --force

Deploy
------

The daily card sender runs on DigitalOcean. To create an app, run:

    doctl apps create --spec .do/app.yaml

Once it is deployed, you can update it with:

    doctl apps update <app id> --spec .do/app.yaml

Note that in both cases, you'll need to add the secrets manually in the UI afterwards.
