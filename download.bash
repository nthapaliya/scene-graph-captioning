#!/bin/bash

# Downloads curated VG150 data
mkdir -p data
wget -nc https://github.com/Maelic/VG_curated/raw/refs/heads/main/VG150_curated.zip -P data
unzip -u -d data data/VG150_curated.zip

cp data/no_part_whole/VG-SGG.h5 data
cp data/no_part_whole/VG-SGG-dicts.json data/

# Downloads updated image_data.json
wget -nc https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/raw/refs/heads/master/datasets/vg/image_data.json -P data


wget -nc http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P data
unzip -u -d data data/annotations_trainval2014.zip
cp data/annotations/captions_train2014.json data/
cp data/annotations/captions_val2014.json data/

# Downloads supplementary files
# wget -nc https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip -P data
# unzip -u -d data data/region_descriptions.json.zip
# wget -nc https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/question_answers.json.zip -P data
# unzip -u -d data data/question_answers.json.zip


# Downloads image files. This take a long time!
wget -nc https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P data
unzip -u -d data data/images.zip
wget -nc https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P data
unzip -u -d data data/images2.zip

# Move images to final folder structure
echo "Moving files to final location"
find data/VG_100K_2 -name '*.jpg' | xargs -J % mv % data/VG_100K/
rm -rf data/VG_100K_2