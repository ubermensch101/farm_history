Structure:
----------

- All relevant constants are stored in `config/`, and are accessed via the Config class.
- Each file must be runnable on its own, and hence must have an if `__name__=="__main__"` section.

Installation flow: always at root
---------------------------------

1. Create a venv using `python3 -m venv venv`
2. Activate venv usin `source venv/bin/activate`
3. `pip3 install -e .` to install everything in requirements.txt

Running the pipeline:
---------------------

- Configuration
  - Open `config/`
  - Configure tables.json (The codebase runs only on the first table in the list)
  - Configure psql.json (postgres database configuration)
- Satellite data
  - All satellite data (quads or otherwise) must be store in `quads/` in the root
  - Run `python3 src/planet_labs/planet_labs.py` to get monthly data of a quad that intersects the table's geometry - Good for small tables
  - Run `python3 src/planet_labs/planet_labs_quadpy` to get specific quads - Better when dealing with a large table (say a taluka)
- Clipping
  - Run `python3 src/crop_presence_basic/clip_automatic.py` to clip satellite quads and store average RGB values in the database as extra columns
- Crop Presence
  - Run `python3 src/crop_presence_basic/crop_presence_inference.py` to infer monthly crop presence probabilities based on the stored average RGB values
  - `python3 src/crop_presence_basic/train_crop_presence.py` reads annotations.csv that must correspond to the existing table being processed (the first table in tables.json) and trains the crop presence model
- Crop Cycle
  - Run `python3 src/crop_cycle_monthly/crop_cycle_inference.py` to infer cropping patterns based on crop probabilities and store the results in the database
  - `python3 src/crop_cycle_monthly/train_crop_cycle.py` trains the crop cyle model based on cycle_annotations.py
- Analysis
  - Run `python3 src/analysis/study_plot.py -k <some_key>` to study the farm plot with that specific key
  - Run `python3 src/analysis/study_random_plot.py` to study a random plot
- CNN Crop Presence
  - Run `python3 src/crop_presence_cnn/train_crop_presence_cnn.py `to train CNN model (Refer to `src/crop_presence_cnn/model.py` for CNN architecture)
  - Run `python3 src/crop_presence_cnn/crop_presence_inference_cnn.py ` to infer monthly crop presence probabilities from histogram model
  - Run `python3 src/crop_presence_cnn/annotate_cnn.py -s <start-key> -e <end-key> -d <train/test> ` to annotate farm plots

Work in Progress:
-----------------

- Switching to a CNN based crop presence predictor
- Shifting to weekly data from planet labs to improve rabi sowing period detection
