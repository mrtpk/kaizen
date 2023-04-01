#!/bin/bash
# Updates mrtpk.github.io
git config --global user.email "bot@mrtpk.github.io"
git config --global user.name "mrtpk"
git config --global credential.helper store
# rm -rf build
# git clone https://github.com/mrtpk/mrtpk.github.io.git build

# echo Publishing following commit:
# git log -1
# git push origin master

cd build
# git submodule update --init --recursive
# git checkout builder
# git submodule update --init --recursive
cd _posts
# git pull origin master
cd ..
# git add _posts
# git commit -m "updated kaizen"
# git push origin builder

bash ./publish.sh

git config --system --unset credential.helper
