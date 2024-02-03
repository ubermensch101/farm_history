The codebase accepts a table as input via config and creates land use output. The current process adds way too many columns to the input table. It's messy, but streamlined.

planet_labs.py: Download tiff files from planet labs (open source)
clip_automatic.py: Clip the tiff files and store average band values corresponding to each farmplot per month
classify.py: A tool to annotate crop presence
trying_ml.py and crop_cycle_dist.py: ML stuff
