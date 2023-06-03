if [[ $# -ne 2 ]]; then
  echo "$0 file_dir_path_of(movielens_1m_dataset.dat) config_file(config.json)"
  exit 1
fi

mkdir -p './outfiles/'
outpath='./outfiles/'

filename="${2##*/}"

python3 -u ./code/main.py -tp $1 -cf $2 > $outpath/$filename.log 2>&1