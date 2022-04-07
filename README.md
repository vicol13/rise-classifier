

## The RISE algorithm

The repository represents the implementation of the RISE algorithm done 
for Supervised Experiential Learning for UPC. 


```diff

📦out                       #logs of the algorithm for dataset
 ┣ 📜cancer.txt
 ┗ 📜wine2.txt

📦rise_classifier           # implementation of classifier 
 ┣ 📂data                   # folder with datadasets
 ┃ ┣ 📜breast-cancer.csv
 ┃ ┣ 📜obesity.csv
 ┃ ┣ 📜seismic-bumps.csv
 ┃ ┗ 📜wine.csv
 ┣ 📂source
 ┃ ┣ 📂rise                 # core logic/classes of the algorithm
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜instance.py
 ┃ ┃ ┣ 📜instance_rule.py
 ┃ ┃ ┣ 📜progress_bar.py
 ┃ ┃ ┣ 📜rise_classifier.py
 ┃ ┃ ┗ 📜rise_utils.py
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜data_utils.py        # utils for data preprocessing
 ┗ ┗ 📜main.py              # entry-point for algorithm
 
 📦documents                # report and rise related documents
 ┣ 📂report
 ┃ ┣ 📂media                # media files for the report
 ┃ ┃ ┣ 📜logo.png
 ┃ ┃ ┗ 📜sel.png
 ┃ ┣ 📂out                  # output of .tex compilier
 ┃ ┃ ┣ 📜 **
 ┃ ┣ 📜refs.bib
 ┃ ┗ 📜valeriu_vicol_rise.tex
 ┣ 📜PW1-SEL-2122.pdf
 ┗ 📜RISE-Domingos-Mahine Learning-24-141-168-1996.pdf
 ```


# Set up the project
1. create virtual environment
```shell
python3 -m venv venv/
```
2. enter virtual environment
```shell
source venv/bin/activate
```
3. install dependencies 
```shell
pip3 install requirements.txt
```
# Running the algorithm

1. enter the virtual environment
```shell
source venv/bin/activate
```

2. run the algorithm
```
python3 rise_classifier/source/main.py <data_set_name>
```
make sure <data_set_name> is in folder rise_classifier/data