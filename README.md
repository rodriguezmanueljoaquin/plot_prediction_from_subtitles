# plot_prediction_from_subtitles

# Create virtual environment
python3 -m venv venv

# Execute environment for Windows
venv\Scripts\activate.bat

# Execute environment for Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# To restore backup
tar xzvf data/movies_preprocessed.tar.gz

# To execute container
docker compose build
docker compose up
