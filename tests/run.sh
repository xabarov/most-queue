#!/usr/bin/env bash

docker build -t most-queue-tester .

docker run most-queue-tester
