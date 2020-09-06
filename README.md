# COVID - 19 Data Science Project	After working data science projects, found that project strucure is important for team communication and project deliver. You may visit [my blog](https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600) for more detail.


This repository is an ongoing COVID - 19 data science project, containing 3 models to predict 	Here is the explantation of folder strucure:
the spread of COVID, and a data visualization of state COVID trends.	- src: Stores source code (python, R etc) which serves multiple scenarios. During data exploration and model training, we have to transform data for particular purpose. We have to use same code to transfer data during online prediction as well. So it better separates code from notebook such that it serves different purpose.

- test: In R&D, data science focus on building model but not make sure everything work well in unexpected scenario. However, it will be a trouble if deploying model to API. Also, test cases guarantee backward compatible issue but it takes time to implement it.

- model: Folder for storing binary (json or other format) file for local use.

- data: Folder for storing subset data for experiments. It includes both raw data and processed data for temporary use.
## Contents	- notebook: Storing all notebooks includeing EDA and modeling stage.


### Data	


- The data used for this project was taken from John Hopkins University and the COVID Tracking Project. Links can be found below	

   - https://covidtracking.com/	

    - https://github.com/CSSEGISandData/COVID-19	


### Models	


- **LSTM:** An LSTM Neural Network model was used to predict global COVID cases.	


- **SIR:** An SIR Epidemic Model was used to predict the number of people that get infected with or recover from COVID in 10 countries with the highest COVID cases.	


- **Gaussian:** A Gaussian Error Function was used to predict US COVID deaths.	



### Visualizations	


**State Visualizations:** Data Visualizations of COVID trends in US States	


---	


Special thanks to Dr. Yusuf Uddin and Mr. Arnan Arefeen from UMKC for their suggestions and feedback 	
for this project.
