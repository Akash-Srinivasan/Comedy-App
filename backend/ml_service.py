import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import redis
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserContext:
    user_id: str
    location_city: str
    location_state: str
    age_range: str
    time_of_day: str
    weather: Optional[str] = None
    day_of_week: int = 0
    is_weekend: bool = False

@dataclass
class ComedianFeatures:
    comedian_id: str
    humor_vector: Dict[str, float]
    popularity_score: float
    performance_metrics: Dict[str, float]
    content_features: Dict[str, float]

class MultiArmedBandit:
    """
    Upper Confidence Bound (UCB) implementation for comedian recommendations
    """
    def __init__(self, redis_client: redis.Redis, exploration_factor: float = 2.0):
        self.redis_client = redis_client
        self.exploration_factor = exploration_factor
        
    def get_comedian_stats(self, user_id: str, comedian_id: str) -> Tuple[float, int]:
        """Get mean reward and number of pulls for a comedian-user pair"""
        key = f"bandit:{user_id}:{comedian_id}"
        data = self.redis_client.hgetall(key)
        
        if not data:
            return 0.0, 0
            
        total_reward = float(data.get(b'total_reward', 0))
        num_pulls = int(data.get(b'num_pulls', 0))
        
        mean_reward = total_reward / num_pulls if num_pulls > 0 else 0.0
        return mean_reward, num_pulls
    
    def select_comedian(self, user_id: str, available_comedians: List[str]) -> str:
        """Select comedian using UCB algorithm"""
        if not available_comedians:
            raise ValueError("No available comedians")
        
        # Get total number of interactions for this user
        total_interactions = 0
        comedian_stats = {}
        
        for comedian_id in available_comedians:
            mean_reward, num_pulls = self.get_comedian_stats(user_id, comedian_id)
            comedian_stats[comedian_id] = (mean_reward, num_pulls)
            total_interactions += num_pulls
        
        # If any comedian hasn't been tried, select it (exploration)
        untried_comedians = [c for c, (_, n) in comedian_stats.items() if n == 0]
        if untried_comedians:
            return np.random.choice(untried_comedians)
        
        # Calculate UCB scores
        ucb_scores = {}
        for comedian_id, (mean_reward, num_pulls) in comedian_stats.items():
            if num_pulls == 0:
                ucb_scores[comedian_id] = float('inf')
            else:
                confidence_bound = np.sqrt(
                    (self.exploration_factor * np.log(total_interactions)) / num_pulls
                )
                ucb_scores[comedian_id] = mean_reward + confidence_bound
        
        # Select comedian with highest UCB score
        selected_comedian = max(ucb_scores, key=ucb_scores.get)
        logger.info(f"UCB selected {selected_comedian} for user {user_id}")
        return selected_comedian
    
    def update_reward(self, user_id: str, comedian_id: str, reward: float):
        """Update reward for comedian-user pair"""
        key = f"bandit:{user_id}:{comedian_id}"
        pipe = self.redis_client.pipeline()
        
        # Get current stats
        current_data = self.redis_client.hgetall(key)
        current_total = float(current_data.get(b'total_reward', 0))
        current_pulls = int(current_data.get(b'num_pulls', 0))
        
        # Update stats
        new_total = current_total + reward
        new_pulls = current_pulls + 1
        
        pipe.hset(key, mapping={
            'total_reward': new_total,
            'num_pulls': new_pulls,
            'last_updated': datetime.now().isoformat()
        })
        pipe.execute()
        
        logger.info(f"Updated bandit: user={user_id}, comedian={comedian_id}, reward={reward}")

class OnlineLearningRecommender:
    """
    Online learning recommendation system with real-time updates
    """
    def __init__(self, redis_client: redis.Redis, learning_rate: float = 0.01):
        self.redis_client = redis_client
        self.learning_rate = learning_rate
        self.embedding_dim = 10
        
    def get_user_embedding(self, user_id: str) -> np.ndarray:
        """Get or initialize user embedding"""
        key = f"embedding:user:{user_id}"
        embedding_str = self.redis_client.get(key)
        
        if embedding_str:
            return np.frombuffer(embedding_str, dtype=np.float32)
        else:
            # Initialize with small random values
            embedding = np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
            self.redis_client.set(key, embedding.tobytes())
            return embedding
    
    def get_comedian_embedding(self, comedian_id: str) -> np.ndarray:
        """Get or initialize comedian embedding"""
        key = f"embedding:comedian:{comedian_id}"
        embedding_str = self.redis_client.get(key)
        
        if embedding_str:
            return np.frombuffer(embedding_str, dtype=np.float32)
        else:
            # Initialize with small random values
            embedding = np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
            self.redis_client.set(key, embedding.tobytes())
            return embedding
    
    def predict_score(self, user_id: str, comedian_id: str) -> float:
        """Predict user-comedian affinity score"""
        user_emb = self.get_user_embedding(user_id)
        comedian_emb = self.get_comedian_embedding(comedian_id)
        return float(np.dot(user_emb, comedian_emb))
    
    def update_embeddings(self, user_id: str, comedian_id: str, 
                         actual_reward: float, context: Optional[UserContext] = None):
        """Update embeddings using gradient descent"""
        user_emb = self.get_user_embedding(user_id)
        comedian_emb = self.get_comedian_embedding(comedian_id)
        
        # Predict current score
        predicted_score = np.dot(user_emb, comedian_emb)
        
        # Calculate error
        error = actual_reward - predicted_score
        
        # Gradient descent updates
        user_grad = error * comedian_emb
        comedian_grad = error * user_emb
        
        # Apply updates with L2 regularization
        regularization = 0.001
        user_emb_new = user_emb + self.learning_rate * (user_grad - regularization * user_emb)
        comedian_emb_new = comedian_emb + self.learning_rate * (comedian_grad - regularization * comedian_emb)
        
        # Normalize embeddings
        user_emb_new = user_emb_new / np.linalg.norm(user_emb_new)
        comedian_emb_new = comedian_emb_new / np.linalg.norm(comedian_emb_new)
        
        # Save updated embeddings
        self.redis_client.set(f"embedding:user:{user_id}", user_emb_new.tobytes())
        self.redis_client.set(f"embedding:comedian:{comedian_id}", comedian_emb_new.tobytes())
        
        logger.info(f"Updated embeddings: user={user_id}, comedian={comedian_id}, error={error:.4f}")

class AdvancedRecommendationEngine:
    """
    Main recommendation engine combining multiple algorithms
    """
    def __init__(self, redis_client: redis.Redis, supabase_client):
        self.redis_client = redis_client
        self.supabase = supabase_client
        self.bandit = MultiArmedBandit(redis_client)
        self.online_learner = OnlineLearningRecommender(redis_client)
        
    def get_contextual_features(self, context: UserContext) -> Dict[str, float]:
        """Extract contextual features for recommendations"""
        features = {}
        
        # Time-based features
        hour = datetime.now().hour
        features['morning'] = 1.0 if 6 <= hour < 12 else 0.0
        features['afternoon'] = 1.0 if 12 <= hour < 18 else 0.0
        features['evening'] = 1.0 if 18 <= hour < 24 else 0.0
        features['late_night'] = 1.0 if 0 <= hour < 6 else 0.0
        
        # Day features
        features['weekend'] = 1.0 if context.is_weekend else 0.0
        features['weekday'] = 1.0 if not context.is_weekend else 0.0
        
        # Age group features
        age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        for group in age_groups:
            features[f'age_{group}'] = 1.0 if context.age_range == group else 0.0
        
        return features
    
    def get_user_interaction_history(self, user_id: str) -> pd.DataFrame:
        """Get user's interaction history from database"""
        try:
            response = self.supabase.table('user_interactions')\
                .select('*, comedians!inner(humor_vector, name)')\
                .eq('user_id', user_id)\
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching user interactions: {e}")
            return pd.DataFrame()
    
    def calculate_similarity_scores(self, user_profile: Dict[str, float], 
                                  comedian_features: List[ComedianFeatures]) -> Dict[str, float]:
        """Calculate content-based similarity scores"""
        scores = {}
        
        for comedian in comedian_features:
            # Calculate cosine similarity between user profile and comedian humor vector
            user_vector = np.array([user_profile.get(dim, 0.5) for dim in comedian.humor_vector.keys()])
            comedian_vector = np.array(list(comedian.humor_vector.values()))
            
            if len(user_vector) > 0 and len(comedian_vector) > 0:
                similarity = cosine_similarity([user_vector], [comedian_vector])[0][0]
                scores[comedian.comedian_id] = similarity
            else:
                scores[comedian.comedian_id] = 0.0
                
        return scores
    
    def get_hybrid_recommendations(self, user_id: str, context: UserContext, 
                                 limit: int = 10) -> List[Dict]:
        """Generate hybrid recommendations using multiple algorithms"""
        # Get user profile
        user_response = self.supabase.table('user_profiles')\
            .select('*')\
            .eq('id', user_id)\
            .execute()
        
        if not user_response.data:
            raise ValueError(f"User profile not found for {user_id}")
        
        user_profile = user_response.data[0]
        
        # Get available comedians
        comedians_response = self.supabase.table('comedians')\
            .select('*')\
            .eq('is_active', True)\
            .execute()
        
        if not comedians_response.data:
            return []
        
        comedian_features = [
            ComedianFeatures(
                comedian_id=c['id'],
                humor_vector=c['humor_vector'],
                popularity_score=c.get('popularity_score', 0.5),
                performance_metrics={},
                content_features={}
            ) for c in comedians_response.data
        ]
        
        # Get comedians user hasn't interacted with
        interaction_history = self.get_user_interaction_history(user_id)
        interacted_ids = set(interaction_history['comedian_id'].tolist()) if not interaction_history.empty else set()
        
        available_comedians = [c for c in comedian_features if c.comedian_id not in interacted_ids]
        available_comedian_ids = [c.comedian_id for c in available_comedians]
        
        if not available_comedian_ids:
            return []
        
        # Method 1: Content-based similarity
        content_scores = self.calculate_similarity_scores(
            user_profile.get('humor_dimensions', {}), 
            available_comedians
        )
        
        # Method 2: Multi-armed bandit selection
        if len(available_comedian_ids) > 1:
            bandit_choice = self.bandit.select_comedian(user_id, available_comedian_ids)
        else:
            bandit_choice = available_comedian_ids[0]
        
        # Method 3: Online learning predictions
        online_scores = {}
        for comedian_id in available_comedian_ids:
            online_scores[comedian_id] = self.online_learner.predict_score(user_id, comedian_id)
        
        # Combine scores with weights
        final_scores = {}
        for comedian_id in available_comedian_ids:
            content_weight = 0.4
            online_weight = 0.4
            bandit_weight = 0.2
            
            content_score = content_scores.get(comedian_id, 0)
            online_score = online_scores.get(comedian_id, 0)
            bandit_bonus = bandit_weight if comedian_id == bandit_choice else 0
            
            final_scores[comedian_id] = (
                content_weight * content_score + 
                online_weight * online_score + 
                bandit_bonus
            )
        
        # Sort and return top recommendations
        sorted_comedians = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for comedian_id, score in sorted_comedians[:limit]:
            comedian_data = next(c for c in comedians_response.data if c['id'] == comedian_id)
            recommendations.append({
                'comedian': comedian_data,
                'score': score,
                'reasons': self._generate_reasons(comedian_data, user_profile, score)
            })
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations
    
    def _generate_reasons(self, comedian: Dict, user_profile: Dict, score: float) -> List[str]:
        """Generate explanation for why comedian was recommended"""
        reasons = []
        
        user_humor = user_profile.get('humor_dimensions', {})
        comedian_humor = comedian.get('humor_vector', {})
        
        # Find top matching dimensions
        matches = []
        for dim, user_val in user_humor.items():
            comedian_val = comedian_humor.get(dim, 0)
            if user_val > 0.6 and comedian_val > 0.6:
                matches.append((dim, min(user_val, comedian_val)))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        
        if matches:
            top_match = matches[0][0].replace('_', ' ').title()
            reasons.append(f"Strong {top_match} comedy match")
        
        if score > 0.8:
            reasons.append("Highly recommended for you")
        elif score > 0.6:
            reasons.append("Good match for your taste")
        
        if comedian.get('popularity_score', 0) > 0.7:
            reasons.append("Popular choice")
            
        return reasons[:3]  # Limit to top 3 reasons
    
    def record_interaction(self, user_id: str, comedian_id: str, 
                          interaction_type: str, rating: Optional[float] = None,
                          context: Optional[UserContext] = None):
        """Record user interaction and update all models"""
        # Convert interaction to reward signal
        reward_mapping = {
            'like': 1.0,
            'save': 1.2,
            'dislike': -0.5,
            'skip': -0.1
        }
        
        base_reward = reward_mapping.get(interaction_type, 0.0)
        
        # Adjust reward based on rating if provided
        if rating is not None:
            rating_factor = (rating - 3.0) / 2.0  # Normalize 1-5 to -1 to 1
            final_reward = base_reward * (1.0 + rating_factor)
        else:
            final_reward = base_reward
        
        # Update bandit
        self.bandit.update_reward(user_id, comedian_id, final_reward)
        
        # Update online learning model
        self.online_learner.update_embeddings(user_id, comedian_id, final_reward, context)
        
        # Store interaction in database
        try:
            self.supabase.table('user_interactions').insert({
                'user_id': user_id,
                'comedian_id': comedian_id,
                'interaction_type': interaction_type,
                'rating': rating
            }).execute()
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
        
        logger.info(f"Recorded interaction: {user_id} -> {comedian_id} ({interaction_type}, reward={final_reward})")

# Example usage and FastAPI integration
if __name__ == "__main__":
    import redis
    from supabase import create_client, Client
    
    # Initialize clients
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    supabase_url = "your_supabase_url"
    supabase_key = "your_supabase_key"
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Initialize recommendation engine
    engine = AdvancedRecommendationEngine(redis_client, supabase)
    
    # Example context
    context = UserContext(
        user_id="test_user",
        location_city="New York",
        location_state="NY",
        age_range="25-34",
        time_of_day="evening",
        is_weekend=False
    )
    
    # Get recommendations
    try:
        recommendations = engine.get_hybrid_recommendations("test_user", context)
        print(f"Generated {len(recommendations)} recommendations")
        for rec in recommendations[:3]:
            print(f"- {rec['comedian']['name']}: {rec['score']:.3f} ({', '.join(rec['reasons'])})")
    except Exception as e:
        print(f"Error: {e}")