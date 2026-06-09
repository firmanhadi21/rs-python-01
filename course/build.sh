#!/bin/bash
# Assembles the course from parts.
# Run from the course directory: bash build.sh
set -e
cat _base.html modules/01-intro.html modules/02-cast.html modules/03-features.html modules/04-pipeline.html modules/05-results.html _footer.html > index.html
echo "Built index.html — open it in your browser."
