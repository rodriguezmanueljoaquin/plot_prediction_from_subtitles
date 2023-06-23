
# To create backups
tar cvzf - data/original_dataset/dataset_True_* | split --bytes=100MB - data/original_dataset/movies_subtitles.tar.gz

# To restore backups
cat data/original_dataset/movies_subtitles.tar.gz.* | tar xzvf -
