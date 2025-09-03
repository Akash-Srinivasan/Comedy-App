# backend/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import redis
import os
from supabase import create_client, Client
from ml_service import AdvancedRecommendationEngine, UserContext
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Comedy Discovery ML API",
    description="Advanced recommendation engine for comedy discovery",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=False  # Keep as bytes for embedding storage
)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_SERVICE_KEY", "")
)

# Initialize recommendation engine
recommendation_engine = AdvancedRecommendationEngine(redis_client, supabase)

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: str
    limit: int = 10
    context: Optional[Dict[str, Any]] = None

class InteractionRequest(BaseModel):
    user_id: str
    comedian_id: str
    interaction_type: str  # 'like', 'dislike', 'save', 'skip'
    rating: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

class ComedianAnalysisRequest(BaseModel):
    comedian_id: str
    content_urls: Optional[List[str]] = None

class UserProfileUpdate(BaseModel):
    user_id: str
    humor_dimensions: Dict[str, float]
    preferences: Dict[str, Any]
    demographic_info: Dict[str, str]

class RecommendationResponse(BaseModel):
    comedian_id: str
    comedian_name: str
    score: float
    reasons: List[str]
    comedian_data: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}
    
    # Check Redis
    try:
        redis_client.ping()
        services["redis"] = "healthy"
    except Exception as e:
        services["redis"] = f"unhealthy: {str(e)}"
    
    # Check Supabase
    try:
        response = supabase.table('comedians').select("count", count="exact").limit(1).execute()
        services["supabase"] = "healthy"
    except Exception as e:
        services["supabase"] = f"unhealthy: {str(e)}"
    
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        timestamp=datetime.now(),
        services=services
    )

@app.post("/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(request: RecommendationRequest):
    """Get personalized comedian recommendations"""
    try:
        # Create user context
        context_data = request.context or {}
        user_context = UserContext(
            user_id=request.user_id,
            location_city=context_data.get("location_city", ""),
            location_state=context_data.get("location_state", ""),
            age_range=context_data.get("age_range", "25-34"),
            time_of_day=context_data.get("time_of_day", "evening"),
            weather=context_data.get("weather"),
            day_of_week=datetime.now().weekday(),
            is_weekend=datetime.now().weekday() >= 5
        )
        
        # Get recommendations from ML engine
        recommendations = recommendation_engine.get_hybrid_recommendations(
            request.user_id, 
            user_context, 
            request.limit
        )
        
        # Format response
        response = []
        for rec in recommendations:
            response.append(RecommendationResponse(
                comedian_id=rec['comedian']['id'],
                comedian_name=rec['comedian']['name'],
                score=rec['score'],
                reasons=rec['reasons'],
                comedian_data=rec['comedian']
            ))
        
        logger.info(f"Served {len(response)} recommendations to user {request.user_id}")
        return {"status": "success", "message": "Model retraining initiated"}
        
    except Exception as e:
        logger.error(f"Error triggering retrain: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_models():
    """Background task to retrain ML models"""
    logger.info("Starting model retraining...")
    # Implementation would go here - train new embeddings, update bandit parameters, etc.
    # This is a placeholder for the actual retraining logic
    logger.info("Model retraining completed")

@app.get("/analytics/performance")
async def get_performance_metrics():
    """Get recommendation system performance metrics"""
    try:
        # Get recent interactions for performance analysis
        from datetime import timedelta
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        
        response = supabase.table('user_interactions')\
            .select('*')\
            .gte('created_at', week_ago)\
            .execute()
        
        interactions = response.data or []
        
        # Calculate metrics
        total_interactions = len(interactions)
        positive_interactions = len([i for i in interactions if i['interaction_type'] in ['like', 'save']])
        
        metrics = {
            "total_interactions_7d": total_interactions,
            "positive_rate_7d": positive_interactions / total_interactions if total_interactions > 0 else 0,
            "interactions_by_type": {},
            "average_rating": None
        }
        
        # Breakdown by interaction type
        for interaction_type in ['like', 'dislike', 'save', 'skip']:
            count = len([i for i in interactions if i['interaction_type'] == interaction_type])
            metrics["interactions_by_type"][interaction_type] = count
        
        # Average rating
        ratings = [i['rating'] for i in interactions if i['rating'] is not None]
        if ratings:
            metrics["average_rating"] = sum(ratings) / len(ratings)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    ) response
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interactions")
async def record_interaction(request: InteractionRequest, background_tasks: BackgroundTasks):
    """Record user interaction with comedian"""
    try:
        # Validate interaction type
        valid_interactions = {'like', 'dislike', 'save', 'skip'}
        if request.interaction_type not in valid_interactions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid interaction type. Must be one of: {valid_interactions}"
            )
        
        # Create user context if provided
        context = None
        if request.context:
            context = UserContext(
                user_id=request.user_id,
                location_city=request.context.get("location_city", ""),
                location_state=request.context.get("location_state", ""),
                age_range=request.context.get("age_range", "25-34"),
                time_of_day=request.context.get("time_of_day", "evening"),
                weather=request.context.get("weather"),
                day_of_week=datetime.now().weekday(),
                is_weekend=datetime.now().weekday() >= 5
            )
        
        # Record interaction in background to avoid blocking
        background_tasks.add_task(
            recommendation_engine.record_interaction,
            request.user_id,
            request.comedian_id,
            request.interaction_type,
            request.rating,
            context
        )
        
        return {"status": "success", "message": "Interaction recorded"}
        
    except Exception as e:
        logger.error(f"Error recording interaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get user interaction statistics"""
    try:
        # Get interaction history
        response = supabase.table('user_interactions')\
            .select('*')\
            .eq('user_id', user_id)\
            .execute()
        
        interactions = response.data or []
        
        # Calculate stats
        stats = {
            "total_interactions": len(interactions),
            "likes": len([i for i in interactions if i['interaction_type'] == 'like']),
            "saves": len([i for i in interactions if i['interaction_type'] == 'save']),
            "dislikes": len([i for i in interactions if i['interaction_type'] == 'dislike']),
            "skips": len([i for i in interactions if i['interaction_type'] == 'skip']),
            "average_rating": None
        }
        
        # Calculate average rating
        ratings = [i['rating'] for i in interactions if i['rating'] is not None]
        if ratings:
            stats["average_rating"] = sum(ratings) / len(ratings)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/comedian/{comedian_id}/similar")
async def get_similar_comedians(comedian_id: str, limit: int = 5):
    """Get comedians similar to the specified comedian"""
    try:
        # Get comedian data
        response = supabase.table('comedians')\
            .select('*')\
            .eq('id', comedian_id)\
            .single()\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Comedian not found")
        
        target_comedian = response.data
        
        # Get all other comedians
        all_comedians_response = supabase.table('comedians')\
            .select('*')\
            .eq('is_active', True)\
            .neq('id', comedian_id)\
            .execute()
        
        all_comedians = all_comedians_response.data or []
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        target_vector = list(target_comedian['humor_vector'].values())
        similarities = []
        
        for comedian in all_comedians:
            comedian_vector = list(comedian['humor_vector'].values())
            if len(target_vector) == len(comedian_vector):
                similarity = cosine_similarity([target_vector], [comedian_vector])[0][0]
                similarities.append({
                    'comedian': comedian,
                    'similarity': float(similarity)
                })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return [
            {
                'comedian_id': sim['comedian']['id'],
                'comedian_name': sim['comedian']['name'],
                'similarity_score': sim['similarity'],
                'comedian_data': sim['comedian']
            }
            for sim in similarities[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Error getting similar comedians: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/retrain")
async def trigger_model_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining (admin endpoint)"""
    try:
        # Add retraining task to background
        background_tasks.add_task(retrain_models)
        
        return