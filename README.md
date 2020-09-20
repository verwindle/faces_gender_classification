<!-- CONTENTS -->
## Contents

* [About the Project](#about-the-project)
  * Task 1. Simple algorithmic script for finding continuous subarray with largest sum.
  * Task 2. Problem of gender classification on dataset of 100000 male and female images. 
* [Usage](#usage)



<!-- ABOUT THE PROJECT -->
## About The Project

![Task Screenshot](https://github.com/verwindle/faces_gender_classification/blob/master/title_sheet.png)

This repository was created for NtechLab CV team verification. And contains my solution for the problem. Description of the problem is clear for russians from the image above.

### Prerequisites

* Linux machine
* Python environment
* Jupyter

This repository contains requirements. Can be installed with.

```sh
pip install -r requirements.txt
```

### Download the dataset

1. Go inside second task
```sh
cd task2
```
2. Modify installation script rights
```sh
chmod +x data_install_script
```
3. Run the script
```sh
./data_install_script
```



<!-- USAGE -->
## Usage

To get json of filenames and predicted gender:

1. If current dir is not "task2"
```sh
cd task2
```
2. Execute script like in the following example:
```sh
python process.py images 100
```
