1. Download dataset: `sh calvin/dataset/download_data.sh D`
2. Generate hdf5 files with language grouping:
```
python tools/convert_calvin_dataset.py -b D --split train --output_name language

python tools/convert_calvin_dataset.py -b D --split val --output_name language
```
3. Generate list of valid indices in that directory, e.g.,
```
ls /mnt/nfs_client/calvin/task_D_D/training/ | grep episode | awk -F '_' '{print $2}' | awk -F '.' '{print $1}' > /mnt/nfs_client/calvin/task_D_D/training/id_list.txt
```
3. Generate missing idxes for that dataset using the file (make sure to change the location) `python scripts/generate_missing_idx.py`
4. Copy the `dataset_config.json` to configs directory and pass it as an argument
