#!/usr/bin/env bash
DATA_PATH=data/drawmeacat
FILTERED_COUNTRY="FR"

mkdir -p $DATA_PATH
cd $DATA_PATH
for class in $(ls data)
    do
    echo $class
    cat data/$class | ndjson-filter "d.recognized == true && d.countrycode == \"$FILTERED_COUNTRY\"" | ndjson-reduce > json/$class.json
done
cd ../..