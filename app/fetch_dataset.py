import os
import sys
sys.path.insert(0, '../src/')

import kaggle
import auto_co2 as co2



def main():
    auth_file_path = sys.argv[1]
    zipfile_path = '../data/automobile-co2-emissions-eu-2021.zip'
    
    co2.data.download_co2_data(auth_file_path=auth_file_path)
    


if __name__ == '__main__':
    main()