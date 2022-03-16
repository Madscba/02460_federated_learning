#!/usr/bin/env bash

# download data and convert to .json format

if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
    cd preprocess
    ./data_to_json.sh
    cd ..
fi
