# receipts_forecasting

Receipts Forecasting
Overview
Receipts Forecasting is an application designed to predict future receipts using various forecasting models. It offers insights through data exploration and renders the model's predictions, allowing users to evaluate and compare the performance of different models.

Live Application
The application is hosted and can be accessed here.

Repository
All the project files are available in this GitHub repository.

Features
Data Exploration: Visualize and explore the dataset used for training the models.
Model Predictions: Evaluate the performance of different forecasting models and compare their predictions.
Future Predictions: Generate and visualize predictions for future receipts using various models.
Models Implemented
LSTM Model
Gradient Boosting Model
Exponential Smoothing State Space Model (ETS)
Running Locally
You can run the model locally using Docker. Instructions for building and running the Docker container are provided in the repository.

sh
Copy code
docker build -t receipts_forecasting .
docker run -p 8501:8501 receipts_forecasting


After running the above commands, open your web browser and navigate to http://localhost:8501 to access the application.

Limitations and Future Work
The application is developed within a constrained timeframe, limiting the refinement of the LSTM model. The results achieved are promising, but there is a scope for enhancement. The limited availability of data and the allocation of 20% of the data for testing impacted the predictive accuracy of the models.

Future work includes refining the existing models, incorporating more sophisticated models like PatchTST from the research paper "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", and extending the dataset for improved accuracy and reliability.

Contributions
Contributions, issues, and feature requests are welcome! Feel free to check issues page.

License
This project is MIT licensed.

