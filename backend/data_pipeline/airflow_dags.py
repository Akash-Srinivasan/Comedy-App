# backend/data_pipeline/airflow_dags.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from feature_engineering import FeatureEngineer
import pandas as pd

default_args = {
    'owner': 'comedy-discovery',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'comedy_feature_pipeline',
    default_args=default_args,
    description='Daily feature engineering pipeline',
    schedule_interval='@daily',
    catchup=False
)

def extract_and_store_features(**context):
    """Extract features and store in feature store"""
    import redis
    from supabase import create_client
    import os
    
    # Initialize clients
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=6379, db=1)
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(supabase, redis_client)
    
    # Get all active users and comedians
    users_response = supabase.table('user_profiles').select('id').execute()
    comedians_response = supabase.table('comedians').select('id').eq('is_active', True).execute()
    
    user_ids = [u['id'] for u in users_response.data]
    comedian_ids = [c['id'] for c in comedians_response.data]
    
    # Create feature matrix
    feature_matrix = feature_engineer.create_feature_matrix(user_ids, comedian_ids)
    
    # Store in Redis with expiration
    feature_key = f"features:{context['ds']}"  # ds is the execution date
    redis_client.set(feature_key, feature_matrix.to_json(), ex=86400*7)  # 7 days expiration
    
    print(f"Extracted features for {len(user_ids)} users and {len(comedian_ids)} comedians")

def update_model_embeddings(**context):
    """Update model embeddings based on recent interactions"""
    # This would update the online learning embeddings
    print("Updated model embeddings")

def calculate_drift_metrics(**context):
    """Calculate data drift metrics"""
    # Implementation for monitoring data drift
    print("Calculated drift metrics")

# Define tasks
feature_extraction_task = PythonOperator(
    task_id='extract_features',
    python_callable=extract_and_store_features,
    dag=dag
)

model_update_task = PythonOperator(
    task_id='update_embeddings',
    python_callable=update_model_embeddings,
    dag=dag
)

drift_monitoring_task = PythonOperator(
    task_id='monitor_drift',
    python_callable=calculate_drift_metrics,
    dag=dag
)

# Set task dependencies
feature_extraction_task >> model_update_task >> drift_monitoring_task