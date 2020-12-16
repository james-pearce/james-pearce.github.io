#!/bin/bash -ex

# Setup git so we can use it
git config --global user.email "jamespearce@3crownsconsulting.com.au"
git config --global user.name "CircleCI build_html script"
# remove changes from current gh-pages-ci branch
git checkout -f
git checkout master

# Make sure that local master matches with remote master
# CircleCI merges made changes to master so need to reset it
git fetch origin master
git reset --hard origin/master

# Gets _site/* files and pushes them to master branch
# Note: CircleCI creates vendor and .bundle files
mv _site /tmp/
rm -rf * .bundle .sass-cache
mv /tmp/_site/* .
git add -A .
git commit -m "CircleCI build_html: copy _site files that was generated from gh-pages-ci branch"
git push origin master
