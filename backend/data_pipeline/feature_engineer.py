# backend/data_pipeline/feature_engineering.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    window_size_days: int = 30
    min_interactions: int = 5
    text_features_max_features: int = 1000

class FeatureEngineer:
    def __init__(self, supabase_client, redis_client, config: FeatureConfig = None):
        self.supabase = supabase_client
        self.redis = redis_client
        self.config = config or FeatureConfig()
        self.scalers = {}
        self.encoders = {}
        
    def extract_user_features(self, user_id: str, as_of_date: datetime = None) -> Dict:
        """Extract comprehensive user features"""
        if as_of_date is None:
            as_of_date = datetime.now()
        
        features = {}
        
        # Get user profile
        user_profile = self._get_user_profile(user_id)
        if not user_profile:
            return {}
        
        # Basic demographic features
        features.update(self._extract_demographic_features(user_profile))
        
        # Interaction-based features
        features.update(self._extract_interaction_features(user_id, as_of_date))
        
        # Temporal behavior features
        features.update(self._extract_temporal_features(user_id, as_of_date))
        
        # Preference consistency features
        features.update(self._extract_consistency_features(user_id, as_of_date))
        
        return features
    
    def _get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile from database"""
        try:
            response = self.supabase.table('user_profiles')\
                .select('*')\
                .eq('id', user_id)\
                .single()\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching user profile {user_id}: {e}")
            return None
    
    def _extract_demographic_features(self, user_profile: Dict) -> Dict:
        """Extract demographic features"""
        features = {}
        
        # Age range encoding
        age_ranges = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        age_range = user_profile.get('age_range', '25-34')
        for age in age_ranges:
            features[f'age_{age}'] = 1.0 if age == age_range else 0.0
        
        # Location features (simplified - in production, use proper geocoding)
        features['location_encoded'] = hash(f"{user_profile.get('location_city', '')}_{user_profile.get('location_state', '')}") % 1000
        
        # Humor dimension features
        humor_dims = user_profile.get('humor_dimensions', {})
        for dim, value in humor_dims.items():
            features[f'humor_{dim}'] = float(value)
        
        # Confidence score
        features['confidence_score'] = float(user_profile.get('confidence_score', 0.1))
        
        return features
    
    def _extract_interaction_features(self, user_id: str, as_of_date: datetime) -> Dict:
        """Extract features from user interaction history"""
        features = {}
        
        # Get interactions within window
        window_start = as_of_date - timedelta(days=self.config.window_size_days)
        
        try:
            response = self.supabase.table('user_interactions')\
                .select('*, comedians!inner(humor_vector, popularity_score)')\
                .eq('user_id', user_id)\
                .gte('created_at', window_start.isoformat())\
                .lte('created_at', as_of_date.isoformat())\
                .execute()
            
            interactions = response.data or []
        except Exception as e:
            logger.error(f"Error fetching interactions for user {user_id}: {e}")
            interactions = []
        
        if len(interactions) < self.config.min_interactions:
            # Not enough data - return default features
            return self._get_default_interaction_features()
        
        # Basic interaction statistics
        features['total_interactions'] = len(interactions)
        features['interactions_per_day'] = len(interactions) / self.config.window_size_days
        
        # Interaction type distribution
        interaction_types = ['like', 'dislike', 'save', 'skip']
        for itype in interaction_types:
            count = sum(1 for i in interactions if i['interaction_type'] == itype)
            features[f'interaction_{itype}_rate'] = count / len(interactions)
        
        # Rating statistics
        ratings = [i['rating'] for i in interactions if i['rating'] is not None]
        if ratings:
            features['avg_rating'] = np.mean(ratings)
            features['rating_std'] = np.std(ratings)
            features['rating_range'] = max(ratings) - min(ratings)
        else:
            features['avg_rating'] = 3.0
            features['rating_std'] = 0.0
            features['rating_range'] = 0.0
        
        # Comedian preference patterns
        features.update(self._extract_comedian_preference_features(interactions))
        
        return features
    
    def _extract_comedian_preference_features(self, interactions: List[Dict]) -> Dict:
        """Extract features about comedian preferences"""
        features = {}
        
        # Get humor vectors for liked comedians
        liked_comedians = [i for i in interactions if i['interaction_type'] in ['like', 'save']]
        
        if not liked_comedians:
            return self._get_default_preference_features()
        
        # Average humor dimensions of liked comedians
        humor_dims = {}
        for interaction in liked_comedians:
            comedian_humor = interaction['comedians']['humor_vector']
            for dim, value in comedian_humor.items():
                if dim not in humor_dims:
                    humor_dims[dim] = []
                humor_dims[dim].append(float(value))
        
        for dim, values in humor_dims.items():
            features[f'liked_comedian_avg_{dim}'] = np.mean(values)
            features[f'liked_comedian_std_{dim}'] = np.std(values) if len(values) > 1 else 0.0
        
        # Popularity preference
        popularity_scores = [float(i['comedians'].get('popularity_score', 0.5)) for i in liked_comedians]
        features['liked_comedian_avg_popularity'] = np.mean(popularity_scores)
        features['likes_popular_comedians'] = np.mean([1 if p > 0.7 else 0 for p in popularity_scores])
        
        return features
    
    def _extract_temporal_features(self, user_id: str, as_of_date: datetime) -> Dict:
        """Extract temporal behavior features"""
        features = {}
        
        try:
            # Get all interactions for temporal analysis
            response = self.supabase.table('user_interactions')\
                .select('created_at, interaction_type')\
                .eq('user_id', user_id)\
                .execute()
            
            interactions = response.data or []
        except Exception as e:
            logger.error(f"Error fetching temporal data for user {user_id}: {e}")
            return self._get_default_temporal_features()
        
        if not interactions:
            return self._get_default_temporal_features()
        
        # Convert timestamps
        timestamps = [pd.to_datetime(i['created_at']) for i in interactions]
        
        # Hour of day patterns
        hours = [ts.hour for ts in timestamps]
        features['most_active_hour'] = max(set(hours), key=hours.count) if hours else 12
        
        # Day of week patterns
        weekdays = [ts.weekday() for ts in timestamps]
        features['most_active_weekday'] = max(set(weekdays), key=weekdays.count) if weekdays else 1
        features['weekend_activity_rate'] = sum(1 for d in weekdays if d >= 5) / len(weekdays) if weekdays else 0.5
        
        # Activity frequency
        if len(timestamps) > 1:
            deltas = [(timestamps[i] - timestamps[i-1]).total_seconds() / 3600 for i in range(1, len(timestamps))]
            features['avg_hours_between_interactions'] = np.mean(deltas)
            features['interaction_frequency_consistency'] = 1.0 / (1.0 + np.std(deltas))
        else:
            features['avg_hours_between_interactions'] = 24.0
            features['interaction_frequency_consistency'] = 0.0
        
        return features
    
    def _extract_consistency_features(self, user_id: str, as_of_date: datetime) -> Dict:
        """Extract features about user preference consistency"""
        features = {}
        
        try:
            response = self.supabase.table('user_interactions')\
                .select('*, comedians!inner(humor_vector)')\
                .eq('user_id', user_id)\
                .execute()
            
            interactions = response.data or []
        except Exception as e:
            logger.error(f"Error fetching consistency data for user {user_id}: {e}")
            return {'preference_consistency': 0.5}
        
        if len(interactions) < 3:
            return {'preference_consistency': 0.0}
        
        # Calculate consistency in humor dimension preferences
        positive_interactions = [i for i in interactions if i['interaction_type'] in ['like', 'save']]
        negative_interactions = [i for i in interactions if i['interaction_type'] in ['dislike', 'skip']]
        
        if not positive_interactions:
            return {'preference_consistency': 0.0}
        
        # Get humor dimensions for positive interactions
        positive_vectors = [i['comedians']['humor_vector'] for i in positive_interactions]
        
        # Calculate average and variance across dimensions
        dimension_consistencies = []
        for dim in positive_vectors[0].keys():
            values = [float(v[dim]) for v in positive_vectors]
            if len(values) > 1:
                consistency = 1.0 - (np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0)
                dimension_consistencies.append(max(0.0, consistency))
        
        features['preference_consistency'] = np.mean(dimension_consistencies) if dimension_consistencies else 0.0
        
        return features
    
    def _get_default_interaction_features(self) -> Dict:
        """Default features for users with insufficient interaction data"""
        return {
            'total_interactions': 0,
            'interactions_per_day': 0.0,
            'interaction_like_rate': 0.0,
            'interaction_dislike_rate': 0.0,
            'interaction_save_rate': 0.0,
            'interaction_skip_rate': 0.0,
            'avg_rating': 3.0,
            'rating_std': 0.0,
            'rating_range': 0.0
        }
    
    def _get_default_preference_features(self) -> Dict:
        """Default preference features"""
        return {
            'liked_comedian_avg_popularity': 0.5,
            'likes_popular_comedians': 0.5
        }
    
    def _get_default_temporal_features(self) -> Dict:
        """Default temporal features"""
        return {
            'most_active_hour': 12,
            'most_active_weekday': 1,
            'weekend_activity_rate': 0.3,
            'avg_hours_between_interactions': 24.0,
            'interaction_frequency_consistency': 0.0
        }
    
    def extract_comedian_features(self, comedian_id: str) -> Dict:
        """Extract comprehensive comedian features"""
        features = {}
        
        try:
            # Get comedian data
            response = self.supabase.table('comedians')\
                .select('*')\
                .eq('id', comedian_id)\
                .single()\
                .execute()
            
            comedian = response.data
            if not comedian:
                return {}
            
            # Basic features
            features['popularity_score'] = float(comedian.get('popularity_score', 0.5))
            
            # Humor vector
            humor_vector = comedian.get('humor_vector', {})
            for dim, value in humor_vector.items():
                features[f'comedian_humor_{dim}'] = float(value)
            
            # Interaction statistics
            features.update(self._extract_comedian_interaction_stats(comedian_id))
            
        except Exception as e:
            logger.error(f"Error extracting features for comedian {comedian_id}: {e}")
        
        return features
    
    def _extract_comedian_interaction_stats(self, comedian_id: str) -> Dict:
        """Extract interaction statistics for a comedian"""
        try:
            response = self.supabase.table('user_interactions')\
                .select('interaction_type, rating')\
                .eq('comedian_id', comedian_id)\
                .execute()
            
            interactions = response.data or []
        except Exception as e:
            logger.error(f"Error fetching comedian interaction stats: {e}")
            return {}
        
        if not interactions:
            return {
                'comedian_total_interactions': 0,
                'comedian_like_rate': 0.0,
                'comedian_avg_rating': 3.0
            }
        
        features = {}
        features['comedian_total_interactions'] = len(interactions)
        
        # Like rate
        likes = sum(1 for i in interactions if i['interaction_type'] in ['like', 'save'])
        features['comedian_like_rate'] = likes / len(interactions)
        
        # Average rating
        ratings = [i['rating'] for i in interactions if i['rating'] is not None]
        features['comedian_avg_rating'] = np.mean(ratings) if ratings else 3.0
        
        return features
    
    def create_feature_matrix(self, user_ids: List[str], comedian_ids: List[str], 
                             as_of_date: datetime = None) -> pd.DataFrame:
        """Create feature matrix for training/inference"""
        if as_of_date is None:
            as_of_date = datetime.now()
        
        feature_rows = []
        
        for user_id in user_ids:
            user_features = self.extract_user_features(user_id, as_of_date)
            
            for comedian_id in comedian_ids:
                comedian_features = self.extract_comedian_features(comedian_id)
                
                # Combine user and comedian features
                combined_features = {
                    'user_id': user_id,
                    'comedian_id': comedian_id,
                    **user_features,
                    **comedian_features
                }
                
                # Add interaction features
                combined_features.update(
                    self._create_interaction_features(user_features, comedian_features)
                )
                
                feature_rows.append(combined_features)
        
        return pd.DataFrame(feature_rows)
    
    def _create_interaction_features(self, user_features: Dict, comedian_features: Dict) -> Dict:
        """Create interaction features between user and comedian"""
        features = {}
        
        # Humor dimension similarities
        user_humor_dims = {k.replace('humor_', ''): v for k, v in user_features.items() if k.startswith('humor_')}
        comedian_humor_dims = {k.replace('comedian_humor_', ''): v for k, v in comedian_features.items() if k.startswith('comedian_humor_')}
        
        similarities = []
        for dim in user_humor_dims:
            if dim in comedian_humor_dims:
                user_val = user_humor_dims[dim]
                comedian_val = comedian_humor_dims[dim]
                similarity = 1.0 - abs(user_val - comedian_val)  # Inverse of absolute difference
                similarities.append(similarity)
                features[f'humor_similarity_{dim}'] = similarity
        
        if similarities:
            features['avg_humor_similarity'] = np.mean(similarities)
            features['max_humor_similarity'] = max(similarities)
            features['min_humor_similarity'] = min(similarities)
        else:
            features['avg_humor_similarity'] = 0.0
            features['max_humor_similarity'] = 0.0
            features['min_humor_similarity'] = 0.0
        
        # Popularity alignment
        user_likes_popular = user_features.get('likes_popular_comedians', 0.5)
        comedian_popularity = comedian_features.get('popularity_score', 0.5)
        features['popularity_alignment'] = user_likes_popular * comedian_popularity
        
        return features


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
    # Implementation depends on your specific model architecture
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