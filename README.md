# car-co2-prediction_pre-release


# **Car CO2 Emissions**

In this project, we aim to build a model capable of predicting automobile CO2 emissions or their CO2 score.
Our dataset has been fetched from the European Environment Agency (EEA).  
It contains information about all car registrations recorded in 2021 in the EU + non EU cooperating countries.  

This repository contains:
- /report: Our project report in text format.
- /streamlit: Our project presentation for the thesis defense.
- /app: Our CLI experimentation app files. This app helps the user fetch the data archive, preprocess country specific datasets, run and save classification / regression models, and record / view experiment data into sqlite tables.
- /notebooks: The jupyter notebooks containing data visualisation, preprocessing steps, classification / regression models and interpretability diagnostics.
- /src/auto_co2: A custom python package containing all the tools we created for fetching the dataset, preprocessing, plotting and displaying tables.
- /data: A sample of the data is available, the dataset will be downloaded and modified in down this folder.
- /database: Contains the database *file experiments.db*. SQL queries can be displayed within the app if needed.
- *setup.py*: File that enables the pip install for the custom package *auto_co2*. Command: **`pip install .`**
- *launcher.py*: The app's launcher file, to execute through command line.
- *Dockerfile*: The app's image builder, allows to run the app in a container without needing any condfiguration.
- *environment.yml*: A simplified version of essential python requirements for conda / anaconda.
- *requirements.txt*: The full list of libraries installed on the python env used for development.

## **1.The App: setup**

The app container is a ready-tu-use development/experimentation environment.  
It contains every python package needed to use our app and our auto_co2 library.  
It also allows to fetch the dataset easily from github.com (git lfs).
It relies on a docker-compose file that **mounts the whole repository as a volume**.  
**This means any file created or modified in the repository during image run will persist after the container has been killed**.
The docker image itself contains a functional, yet barebone version of the repository,  
the following files need to be generated using the app:
- Datasets raw and processed
- Machine learning / deep learning trained models
- Interpretability models


### **A. Run in a docker container**


#### **Windows / Mac with Docker Desktop**

- Open Docker Desktop: Start Docker Desktop from the start menu or the system tray.
- Open the Docker Dashboard: Click on the Docker icon in the system tray and select 'Dashboard' from the context menu.
- Navigate to the Images section: In the Docker Dashboard, click on the 'Images' tab on the left side.
- Build the Docker image: 
    - Click on the 'Build...' button at the top right of the Docker Dashboard. A dialog box will open. 
    - In the 'Build context' field, browse to the directory of your Dockerfile (the cloned repository). 
    - In the 'Dockerfile' field, select the Dockerfile. 
    - In the 'Tag' field, enter the name you want to give to your Docker image. 
    - Click on the 'Build' button to start building the image.

Locate the Compose File: Identify the directory containing the provided Docker Compose file ('docker-compose.yml' in this case).   
This file defines the container configuration and dependencies for your application.

Launch Docker Desktop: Launch Docker Desktop from your Windows Start Menu or system tray.  
Wait for Docker to start and initialize properly.

Open Docker Desktop GUI: Switch to the Docker Desktop GUI.  
You can find it in the Applications menu or pinned to your taskbar.

Import Compose File: Click on the "Compose" icon in the top right corner of the Docker Desktop window.  
Choose "Import Compose file" and navigate to the directory containing your 'docker-compose.yml' file. Select the file and click "Open."

Start the Application: Click the "Play" button next to the "repo" service to start the interactive container.  
This will run the Python CLI app for machine learning experiments.

Access the Application: Once the container is running, you can access the application from within the container.  
The Compose file mounts the entire repository to the container's '/app' directory, so any changes you make to the code will be persistent.

Interact with the Application: Within Docker Desktop's GUI, you can attach to the running container by clicking on the "Attach" button next to the "repo" service.  
This will open a terminal window inside the container where you can execute commands, including running the Python CLI app.


#### **Linux / bash terminal**

**Build the image: the easy way...**
- Have Docker Engine running
- Start the app with `python launcher.py`
- Press [1]

**Build the image: the right way**
- Open a terimnal session in the repository root
- Build the image with `docker build -t auto_co2 .`

**Start the image with docker-compose**
- Open a terminal session in the repository root
- Type `docker-compose run repo`
- The app starts in a container in interactive mode
- Exit the app and type `docker-compose down``


### **B. Run the app with a dedicated conda environment**

- Open a terminal session in the repository root
- Type `conda env create -f  environment.yml`
- You'll also need to install sqlite3 to process database actions
- Activate your conda "co2" environment: `conda activate co2`
- run `python launcher.py` 



## **2. The App: guide**

Here's the main menu:

![Alt text](readme/image.png)

- Press 2 + ENTER to download the dataset from the github's repo page

- Press 3 + ENTER to run preprocessing. You'll be prompted for options: select countries, etc.  

![Alt text](<readme/Screenshot from 2024-01-25 17-07-27.png>)  
![Alt text](<readme/Screenshot from 2024-01-25 17-08-06.png>)
![Alt text](<readme/Screenshot from 2024-01-25 17-08-20.png>)  

- Press 4 + ENTER to run classification or press 5 + ENTER to run regression. You'll be prompted for hyperparameter selection, etc.

![Alt text](<readme/Screenshot from 2024-01-25 16-57-56.png>)

- Once the model has been trained, it is evaluated on the test set:  
![Alt text](<readme/Screenshot from 2024-01-25 16-59-31.png>)

- You are given the choice to save the model and write the results in the database (/database/experiments.db)

- Press 8 + ENTER to run SQL queries and explore the experimentation results:  

![Alt text](<readme/Screenshot from 2024-01-25 17-00-41.png>)  



## **Guide to the auto_co2 python package**

This package consists in 4 submodules:
- **Data**: A set of tools created to load, preprocess and save content from the dataset
- **Agg**: Aggregators to help analyze data, draw tables and plot figures on country level, manufacturer level,  car model level
- **styles**: Tools to take advantage of pandas stylers and pretty print pandas objects
- **viz**: Globally all plotting that we used but translated into Plotly Express / Plotly Graph Objects for esthetics and interactivity


### **A. Agg**

Some functionalities:


Instance of the country aggregator object
![Alt text](<readme/Screenshot from 2024-01-25 16-29-22.png>)
![Alt text](<readme/Screenshot from 2024-01-25 16-35-21.png>)


Instance of the manufacturer aggregator object, and sorted display:  
![Alt text](<readme/Screenshot from 2024-01-25 16-29-58.png>)

Manufacturer aggregator's plot_popular_pool_brands:  
![Alt text](<readme/Screenshot from 2024-01-25 16-30-21.png>)



Instance of the car aggregator object
![Alt text](<readme/Screenshot from 2024-01-25 16-30-58.png>)

Car aggregator's spec method:  
![Alt text](<readme/Screenshot from 2024-01-25 16-32-16.png>)
![Alt text](<readme/Screenshot from 2024-01-25 16-32-31.png>)


Car aggregator's polar chart method:
![Alt text](<readme/Screenshot from 2024-01-25 16-32-57.png>)

Car aggregator's facetplot method (bubble = car model, bubble size = count, color = car brand, facetplot = car pool (consolidated group))




### **B. Viz**

#### **Preprocessing**

![Alt text](readme/image-4.png)  

![Alt text](readme/image-5.png)  

![Alt text](readme/image-6.png)

![Alt text](readme/image-7.png)

#### **Machine Learning**

![Alt text](readme/image-11.png)  

![Alt text](readme/image-12.png)  

![Alt text](readme/image-13.png)  

![Alt text](readme/image-14.png)  

![Alt text](readme/image-16.png)


### **Styles**

![Alt text](readme/image-1.png)

![Alt text](readme/image-2.png)

![Alt text](readme/image-3.png)


#### **Machine Learning**

![Alt text](readme/image-8.png)  

![Alt text](readme/image-9.png)  

![Alt text](readme/image-10.png)  

![Alt text](readme/image-15.png)
