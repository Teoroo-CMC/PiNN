#!/bin/bash

# SHELL script for building the documentation and push to github pages
# taken from here: https://www.alkaline-ml.com/2018-12-23-automate-gh-builds/

set -e

cd doc
make clean html
cd ..

mv doc/_build/html ./
git stash
git checkout --orphan gh-pages-tmp
git config --global user.email "$GH_EMAIL" > /dev/null 2>&1
git config --global user.name "$GH_NAME" > /dev/null 2>&1
shopt -s extglob
rm -r !(".git"|"html"|".."|".")
touch .nojekyll
mv html/* ./
rm -r html/

if [[ "$CIRCLE_BRANCH" =~ ^dev$|^[0-9]+\.[0-9]+\.X$ ]]; then
    git add --all
    git commit --allow-empty  -m "Publishing updated documentation..."
    git remote rm origin
    git remote add origin https://"$GH_NAME":"$GH_TOKEN"@github.com/teoroo-cmc/PiNN_dev.git
    git push --force origin gh-pages-tmp:gh-pages
else
    echo "Not on dev, so won't push doc"
fi

