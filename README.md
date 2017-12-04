# kaggle-titanic

Working on [kaggle titanic problem](https://www.kaggle.com/c/titanic) 

 - data goes into /data
 - code goes into /bin
 
## General notes
`generateDatasets.py` will take the CSV input files and generate the datasets that will be used later with the models.  Once processed and features have been added to the datasets the objects are pickled (`pickle.dump()`) and stored in the `/data` directory.  

`modelDatasets.py` loads the pickled data sets (`pickle.load()`) from the `data` directory and then applies models to them.

The binary file created by this script will not be uploaded to github, so run `./generateDatasets.py` before `./modelDatasets.py` for best results.

04/12/2017 - JON - corrected bug in modelDatasets.py and tested push of changes to Github.
