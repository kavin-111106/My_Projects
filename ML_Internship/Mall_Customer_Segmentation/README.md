## Mall Customer Segmentation using K-Means Clustering

## Project Overview
This project performs customer segmentation using the K-Means clustering algorithm on the Mall Customer Segmentation dataset. The goal is to group customers based on their annual income and spending score, helping businesses understand customer groups for targeted marketing strategies.

## Dataset
- Dataset Name: Mall Customer Segmentation Data
- Source: [Kaggle Dataset Link](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- File: `data/Mall_Customers.csv`

## Project Structure
Mall-Customer-Segmentation/
data -- Mall_Customers.csv ||
mall_customer_segmentation.ipynb ||
README.md ||
requirements.txt



## Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Steps Performed
- Data loading and inspection
- Feature selection: Annual Income and Spending Score
- Elbow method to determine the optimal number of clusters
- Applying K-Means clustering algorithm
- Visualizing customer segments using scatter plots

## Output
- Elbow Plot: helps decide the optimal number of clusters
- Customer Segments Plot: visualizes clusters in terms of income vs spending

## How to Run
1. Clone this repository or download the ZIP.
2. Navigate to the project directory:
cd Mall-Customer-Segmentation
3. Install required Python libraries:
pip install -r requirements.txt
4. Run the Jupyter Notebook:
jupyter notebook
5. Open `mall_customer_segmentation.ipynb` and run the cells to see outputs and plots.

## Requirements
pandas
matplotlib
seaborn
scikit-learn

pgsql
Copy
Edit

## Conclusion
Using K-Means clustering, we can successfully segment mall customers into distinct groups based on their spending habits and annual income. This segmentation can help businesses make better decisions regarding marketing strategies.
