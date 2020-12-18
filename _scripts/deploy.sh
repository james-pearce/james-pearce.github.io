#! bin/bash
# _scripts/deploy.sh

DEPLOY_DIR=$(mktemp -d)
BUILD_DIR=$(readlink -f "./_site")

# Clone target repo to a temp directory to ensure no name collision
git clone $CIRCLE_REPOSITORY_URL $DEPLOY_DIR
cd $DEPLOY_DIR

# verify clean repo
git fetch origin
git checkout master

# configure git committer
git config --global user.name CI
git config --global user.email pearce.je@icloud.com

# copy compiled site from artifacts
rsync -a --delete --exclude=.git BUILD_DIR/ .

# commit changes
git add -A
git commit -m "Deployed in CI: $CIRCLE_BUILDD_URL"
git push origin master

# ping search engines with the new sitemap
# curl "http://www.google.com/webmasters/sitemaps/ping?sitemap=https://www.example.com/sitemap.xml"
