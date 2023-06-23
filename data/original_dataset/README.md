
# To create backups
tar cvzf - dataset/dataset_True_* | split --bytes=100MB - dataset/movies_subtitles.tar.gz

# To restore backups
cat movies_subtitles.tar.gz.* | tar xzvf -
