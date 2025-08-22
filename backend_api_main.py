# backend/api/main.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from pydantic import BaseModel
import logging

from .models.credit_scoring import CreditScoringEngine
from .data_pipeline.data_ingestion import DataIngestionManager
from .explainability.causal_engine import CausalExplanationEngine
from .utils.websocket_manager import ConnectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CredScope AI - Real-Time Credit Intelligence",
    description="Advanced explainable credit scoring platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
credit_engine = CreditScoringEngine()
data_manager = DataIngestionManager()
causal_engine = CausalExplanationEngine()
websocket_manager = ConnectionManager()

# Data models
class CreditScoreRequest(BaseModel):
    entity_id: str
    entity_type: str  # "company", "sovereign", "asset"
    include_explanation: bool = True

class ScenarioAnalysis(BaseModel):
    entity_id: str
    scenario_changes: Dict[str, float]
    time_horizon: int = 30

class CreditScoreResponse(BaseModel):
    entity_id: str
    score: float
    rating: str
    confidence: float
    timestamp: datetime
    explanation: Dict[str, Any]
    trends: Dict[str, Any]
    alerts: List[str]

# API Routes
@app.get("/")
async def root():
    return {"message": "CredScope AI - Real-Time Credit Intelligence Platform"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/entities")
async def get_tracked_entities():
    """Get list of all tracked entities"""
    entities = await data_manager.get_tracked_entities()
    return {"entities": entities}

@app.get("/score/{entity_id}")
async def get_credit_score(entity_id: str, include_explanation: bool = True):
    """Get current credit score for an entity"""
    try:
        # Get latest data for entity
        entity_data = await data_manager.get_entity_data(entity_id)
        if not entity_data:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        # Calculate credit score
        score_result = await credit_engine.calculate_score(entity_data)
        
        # Generate explanation if requested
        explanation = {}
        if include_explanation:
            explanation = await causal_engine.generate_explanation(
                entity_data, score_result
            )
        
        # Get trends and alerts
        trends = await credit_engine.get_trends(entity_id)
        alerts = await credit_engine.get_alerts(entity_id)
        
        response = CreditScoreResponse(
            entity_id=entity_id,
            score=score_result['score'],
            rating=score_result['rating'],
            confidence=score_result['confidence'],
            timestamp=datetime.utcnow(),
            explanation=explanation,
            trends=trends,
            alerts=alerts
        )
        
        # Broadcast update via WebSocket
        await websocket_manager.broadcast_update(entity_id, response.dict())
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating score for {entity_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scores/batch")
async def get_batch_scores(entity_ids: List[str]):
    """Get scores for multiple entities"""
    results = []
    for entity_id in entity_ids:
        try:
            score_response = await get_credit_score(entity_id)
            results.append(score_response)
        except HTTPException:
            results.append({"entity_id": entity_id, "error": "Not found"})
    
    return {"scores": results}

@app.get("/historical/{entity_id}")
async def get_historical_scores(
    entity_id: str,
    days: int = 30,
    granularity: str = "daily"
):
    """Get historical score data"""
    try:
        historical_data = await credit_engine.get_historical_scores(
            entity_id, days, granularity
        )
        return {"entity_id": entity_id, "historical_data": historical_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scenario-analysis")
async def run_scenario_analysis(scenario: ScenarioAnalysis):
    """Run what-if scenario analysis"""
    try:
        # Get current entity data
        entity_data = await data_manager.get_entity_data(scenario.entity_id)
        
        # Apply scenario changes
        modified_data = entity_data.copy()
        for feature, change in scenario.scenario_changes.items():
            if feature in modified_data:
                modified_data[feature] *= (1 + change)
        
        # Calculate new score
        new_score = await credit_engine.calculate_score(modified_data)
        original_score = await credit_engine.calculate_score(entity_data)
        
        # Generate impact analysis
        impact_analysis = await causal_engine.analyze_scenario_impact(
            original_data=entity_data,
            modified_data=modified_data,
            original_score=original_score,
            new_score=new_score
        )
        
        return {
            "scenario": scenario.dict(),
            "original_score": original_score,
            "new_score": new_score,
            "impact_analysis": impact_analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-overview")
async def get_market_overview():
    """Get overall market risk overview"""
    try:
        overview = await credit_engine.get_market_overview()
        return overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sector-analysis/{sector}")
async def get_sector_analysis(sector: str):
    """Get sector-specific risk analysis"""
    try:
        analysis = await credit_engine.get_sector_analysis(sector)
        return {"sector": sector, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news-impact/{entity_id}")
async def get_news_impact(entity_id: str, hours: int = 24):
    """Get recent news impact analysis"""
    try:
        news_data = await data_manager.get_recent_news(entity_id, hours)
        impact_analysis = await causal_engine.analyze_news_impact(
            entity_id, news_data
        )
        return {
            "entity_id": entity_id,
            "time_window_hours": hours,
            "news_impact": impact_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                entity_ids = message.get("entity_ids", [])
                await websocket_manager.subscribe_to_entities(client_id, entity_ids)
                
            elif message.get("type") == "unsubscribe":
                entity_ids = message.get("entity_ids", [])
                await websocket_manager.unsubscribe_from_entities(client_id, entity_ids)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)

# Background task for continuous data processing
@app.on_event("startup")
async def startup_event():
    """Initialize background tasks"""
    logger.info("Starting CredScope AI Platform...")
    
    # Start data ingestion
    asyncio.create_task(data_manager.start_continuous_ingestion())
    
    # Start real-time scoring updates
    asyncio.create_task(credit_engine.start_continuous_scoring())
    
    logger.info("Platform started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down CredScope AI Platform...")
    await data_manager.stop_ingestion()
    await credit_engine.stop_scoring()

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
