#!/bin/bash
# Assembles the Indonesian course from parts.
# Run from the course directory: bash build-id.sh
set -e
cat _base-id.html modules/01-intro-id.html modules/02-cast-id.html modules/03-features-id.html modules/04-pipeline-id.html modules/05-results-id.html _footer.html > index-id.html
echo "Built index-id.html — open it in your browser."
