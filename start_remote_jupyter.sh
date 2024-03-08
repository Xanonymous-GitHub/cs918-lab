#!/usr/bin/env bash

./setup.sh && \
poetry run jupyter lab --no-browser --port=8888
