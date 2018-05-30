# LSUN Database (Room images)

- Data Download: [link](https://github.com/fyu/lsun)

- You can get lmdb data

- lmdb to png conversion

```bash
activate python2
python lmdb2img.py convert <lmdb_dir> --out_dir <output_dir>
```

- Create Filelist

```bash
cd <database_dir>
dir /b /s > filelist.txt
```
- You can set the filelist name other than filelist.txt if you want.
- In filelist.txt, delete line: your_database_dir/filelist.txt