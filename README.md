INSTRUCTIONS

1. Install dependencies (assuming pip3 and python3 is installed) 
	[[tesnsorflow],
	 [pandas],
	 [transformers],
	 [numpy],
	 [datetime],
	 [random]]
	 
2. train.csv and test.csv files must be in the same folder as other *.py files

3. download the checkpoints folder (can be found in https://drive.google.com/drive/folders/1Wefc3I2tGoV5SUQNFRy7bxpyMUJyncJq?usp=sharing) and put it in the same directory, together with *.csv and *.py files 

3. Run code to reproduce main results 
	python3 test.py
	
 The test.py file evaluates the test data using the previously saved weights and everytime randomly selects an evidence pair from validation data set and predicts the probability of convincingness. The evidence pairs and the final result are printed correspondingly.
