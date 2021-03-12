# Prepare data

## COCO
Please download coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
and coco images from [2014 training images](http://images.cocodataset.org/zips/train2014.zip) and [2014 val images](http://images.cocodataset.org/zips/val2014.zip).

## Im2Flow
You can set up Im2Flow from authors page [link](https://github.com/rhgao/Im2Flow). Then, you can apply Im2Flow to coco 2014 training images and 2014 val images.

## Bottomup
Please download bottomup features from [link](https://github.com/peteanderson80/bottom-up-attention). We used the 36 features per image (fiexed). Next, unzip the folder and place unzipped folder in 'bottom-up_features' folder. Then, using ./bottom-up_features/tsv.py script(need python2). The script create HDF5 and PKL files. Move these created files to the folder 'final_dataset'. More details about bottomup preparation can be found in [link](https://github.com/poojahira/image-captioning-bottom-up-top-down)

For example:
```
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
unzip trainval_36.zip
cd trainval_36
mv * ../bottom-up_features
cd ..
python2 ./bottom-up_features/tsv.py
mkdir final_dataset
mv ./bottom-up_featurs/"created HDF5 and PKL files" ./final_dataset
```

## Create final datasets
Next, you can create datasets with above datas using ./crate_input_files.py script(need python3). Please modified "data_paths" in create_input_files.py and make_datasets.py.
Note: For object detection, we use maskrcnn_benchmark from [link](https://github.com/facebookresearch/maskrcnn-benchmark).

For example:
```
python3 create_input_files.py

```