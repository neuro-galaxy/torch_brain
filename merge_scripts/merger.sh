#!/usr/bin/env bash

set -e
ROOT=`pwd`

echo $ROOT

rm -rf brainsets temporaldata torch_brain
pip install git-filter-repo

squash_old() {
    # Squashes all commits before $COMMIT into one single commit
    # The commit message is taken from ${REPO_NAME}_commit.txt
    # Commit history beyond $COMMIT is preserved
    REPO_NAME=$1
    COMMIT=$2

    cd $ROOT/$REPO_NAME

    git checkout --orphan squashed-base $COMMIT
    git commit --author "Mehdi Azabou <36160333+mazabou@users.noreply.github.com>" -F $ROOT/${REPO_NAME}_commit.txt
    git rebase --onto squashed-base $COMMIT main
    git branch -D squashed-base

    git filter-repo --commit-callback '
        commit.committer_name = commit.author_name
        commit.committer_email = commit.author_email
        commit.committer_date = commit.author_date
    ' --force
}

to_subdirectory() {
    # Moves all files in the repo to a subdirectory with the repo name
    # Commit history is preserved
    REPO_NAME=$1

    cd $ROOT/$REPO_NAME
    git filter-repo --to-subdirectory-filter $REPO_NAME/
    git filter-repo --message-callback '
        import re
        return re.sub(rb"#(\d+)", rb"'"${REPO_NAME}"'#\1", message)
    '
}

merge() {
    REPO_NAME=$1
    git remote add $REPO_NAME ../$REPO_NAME
    git fetch $REPO_NAME
    git merge $REPO_NAME/main --allow-unrelated-histories -m "Merge $REPO_NAME"
    git remote remove $REPO_NAME
}

git clone git@github.com:neuro-galaxy/temporaldata.git
git clone git@github.com:neuro-galaxy/brainsets.git
git clone git@github.com:neuro-galaxy/torch_brain.git

squash_old brainsets 7649d11
squash_old temporaldata a0aea7f
squash_old torch_brain 81015c7

to_subdirectory brainsets
to_subdirectory temporaldata

cd $ROOT/torch_brain
git checkout -b vinam/merger
merge temporaldata
merge brainsets
