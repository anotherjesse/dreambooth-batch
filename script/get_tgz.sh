#!/bin/sh

# Prep cog with NSFW safety weights before using cog to build container.

# see DOWNLOAD variables section for steps to build a new tarball

# NOTE: this script provides a cachable way of building the container with weights.
# As the builder might be run on ubuntu, or alpine, or any other linux distro, this 
# script should be written to minimize dependencies on the host system.

# Current dependencies:
# - curl
# - md5sum
# - tar
# - CACHE_DIR (passed as first argument)

set -o xtrace  # enable command tracing
set -o errexit # exit on error

# This script is meant to live in the script directory of a cog repo
# To test, test that cog.yaml exists in the parent directory of this script

SCRIPT_DIR=$(dirname "$0")
PARENT_DIR=$(dirname "$SCRIPT_DIR")


# DOWNLOAD variables

# The tarball contains the safety weights, which are downloaded to `diffusers-cache`
# directory via the `script/download-weights` script.

# To create a new tarball, run that script, then create and update a new tarball
# with the following command:

#    script/download-weights
#    tar -cf safety.tar diffusers-cache/
#    md5sum safety.tar
#    mv safety.tar safety-<md5sum>.tar
#    gsutil cp safety-<md5sum>.tar gs://replicant-misc

# then update the DOWNLOAD variables below with the new md5sum and url

DOWNLOAD_URL="$1"
OUTPUT_DIR="$PARENT_DIR/weights"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# download with continue / this should be a no-op if the file already exists
curl "$DOWNLOAD_URL" | tar -xzf - -C "$OUTPUT_DIR"
