# plot_prediction_from_subtitles

# Create virtual environment
python3 -m venv venv

# Execute environment for Windows
venv\Scripts\activate.bat

# Execute environment for Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# To create backups
tar cvzf - dataset/dataset_True_* | split --bytes=100MB - dataset/movies_preprocessed.tar.gz

# To restore backups
cat movies_subtitles.tar.gz | tar xzvf -
