# Laptop Price Prediction for the Sri Lankan Market

### Note
> Note: We can easily run the Streamlit app without putting it into the Docker container. However, in order to gain some experience with Docker, I have containerized it.

## Overview
This project aims to predict the prices of second-hand laptops in the Sri Lankan market using machine learning techniques. By utilizing a fine-tuned Random Forest Regressor model, we estimate prices based on a set of hardware and software features.

The project is implemented following a structured workflow, from data preparation to deployment, and is deployed as a Streamlit web application for user interaction.

## Problem Description
Pricing second-hand laptops accurately is crucial for buyers and sellers in the Sri Lankan market. However, market prices vary significantly based on brand, specifications, and other features. The objective of this project is to create a model that predicts laptop prices using the following features: <br>
   *  __Laptop Brand__, __RAM Size__, __CPU Company__, __CPU Frequency__, __Storage Type__, __GPU Company__, __Category__, __Operating System (OS)__, __Resolution__, __Processor Generation__, __Storage Size__ <br>
   
By analyzing these features, the model can provide price estimates to aid decision-making in this niche market.

## Dataset
The dataset used for training can be found on Kaggle: [dataset](https://www.kaggle.com/datasets/owm4096/laptop-prices)

## Running the Application with Docker
Step 1: Clone this repo: <br>
   `git clone https://github.com/haniyrasul/mlzoomcamp-midtermproject/laptop-price-prediction.git` <br>
   `cd laptop-price-prediction` <br>

Step 2: Build the Docker Image <br>
   `docker build -t laptop-price-predictor .` <br>

Step 3: Run the Docker Container <br>
   `docker run -it --rm -p 9090:9090 laptop-price-predictor` <br>

Step 4: Access the Applicaiton <br>
   `http://localhost:9090`

## Deployed Model: 
  Streamlit Application: [link](https://mlzoomcamp-midtermproject-rqxt4cg2zxhhegbsfmmtsm.streamlit.app/)
