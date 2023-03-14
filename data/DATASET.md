## Dataset Description

The dataset used for this demo is a collection of wikipedia information about various people from https://www.kaggle.com/datasets/sameersmahajan/people-wikipedia-data.  
In the dataset, presented as a CSV file, each row corresponds to a person and the wikipedia entry of them. 
The original dataset consists of three features `URI`, `name` and `text`

For the reference solution, the data is only extracted from the `text` feature.
> text: Provides the text from the Wikipedia page of the corresponding person.

### Setting Up the Data

The benchmarking scripts expects the CSV file to be present in `data/` directory

#### Step 1
Create a kaggle folder using 'mkdir .kaggle' in root directory only 
Navigate inside the kaggle folder using the command 'cd .kaggle'
Install kaggle if not done using the below command: pip install kaggle
Login to Kaggle account. Go to 'Account Tab' & select 'Create a new API token'. This will trigger the download of kaggle.json file in your local system.
This file contains your API credentials and place it in VM  
Move the downloaded 'kaggle.json' file to folder '.kaggle'. 
For example -[mv /home/username/kaggle.json  /home/azureuser/.kaggle]
Make sure kaggle.json should be in <root directory>/.kaggle/kaggle.json [Supported format ]

> Warning : If you get any error like  please check your proxy settings , please configure accordingly by adding the following lines to bashrc files

```sh
export KAGGLE_USERNAME=''
export KAGGLE_KEY=''
export KAGGLE_PROXY="http://

```
You can also refer to this link : https://github.com/Kaggle/kaggle-api/issues/6

#### Step 2
To setup the data for benchmarking under these requirements, run the following set of commands from the 
`data` directory.  

Navigate inside the data folder 'cd data'

```shell
kaggle datasets download sameersmahajan/people-wikipedia-data
unzip people-wikipedia-data.zip -d people-wikipedia-data
mv people-wikipedia-data/people_wiki.csv .
```

> Npte: A kaggle account is necessary to use the kaggle CLI.  Instructions can be found at https://github.com/Kaggle/kaggle-api.

> **Please see this data set's applicable license for terms and conditions. Intel Corporation does not own the rights to this data set and does not confer any rights to it.**