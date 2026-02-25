#!/bin/bash

MANIFEST="gdc_manifest.2026-02-25.041846.txt"
OUTDIR="tcga_gbm_lgg_pedi"

# Extract UUIDs (skip header)
cut -f1 $MANIFEST | tail -n +2 > uuids.txt

mkdir -p $OUTDIR
cd $OUTDIR

echo "Starting downloads..."

cat ../uuids.txt | xargs -n 1 -P 8 -I {} \
curl -L -OJ https://api.gdc.cancer.gov/data/{}

echo "Downloads complete."
