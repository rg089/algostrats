CONFIG_FILE=$1
python rlagents_train.py -c ${CONFIG_FILE}
python rlagents_test.py -c ${CONFIG_FILE}
