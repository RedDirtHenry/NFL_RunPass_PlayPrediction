from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__=="__main__":
    # Run Pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/home/jtotiker/Documents/DataScience/Projects/MLOps/Data/play_by_play_2022.csv")

# mlflow ui --backend-store-uri "file:/home/jtotiker/.var/app/com.visualstudio.code/config/zenml/local_stores/ac87a525-76d3-47fc-9346-719678747e75/mlruns"