# backend/models/credit_scoring.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import asyncio
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for modeling entity relationships"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(GraphNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph convolution layers
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # Apply graph convolutions
        h1 = self.activation(torch.mm(adj_matrix, torch.mm(x, self.gc1.weight.t()) + self.gc1.bias))
        h1 = self.dropout(h1)
        
        h2 = self.activation(torch.mm(adj_matrix, torch.mm(h1, self.gc2.weight.t()) + self.gc2.bias))
        h2 = self.dropout(h2)
        
        output = torch.mm(adj_matrix, torch.mm(h2, self.gc3.weight.t()) + self.gc3.bias)
        
        return torch.sigmoid(output)

class AttentionMechanism(nn.Module):
    """Attention mechanism for feature importance"""
    
    def __init__(self, input_dim: int):
        super(AttentionMechanism, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate attention weights
        attention_scores = torch.softmax(self.attention_weights(x), dim=0)
        
        # Apply attention
        attended_features = x * attention_scores
        
        return attended_features, attention_scores.squeeze()

class CreditScoringEnsemble(nn.Module):
    """Ensemble model combining multiple approaches"""
    
    def __init__(self, feature_dim: int):
        super(CreditScoringEnsemble, self).__init__()
        self.feature_dim = feature_dim
        
        # Neural network component
        self.nn_layers = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism
        self.attention = AttentionMechanism(feature_dim)
        
        # Graph neural network
        self.gnn = GraphNeuralNetwork(feature_dim, 128, 1)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Neural network prediction
        nn_output = self.nn_layers(x)
        
        # Attention-weighted features
        attended_features, attention_weights = self.attention(x)
        attended_output = self.nn_layers(attended_features)
        
        # Graph neural network prediction (if adjacency matrix provided)
        if adj_matrix is not None:
            gnn_output = self.gnn(x.unsqueeze(0), adj_matrix).squeeze()
        else:
            gnn_output = nn_output
        
        # Ensemble prediction
        ensemble_output = (
            self.ensemble_weights[0] * nn_output +
            self.ensemble_weights[1] * attended_output +
            self.ensemble_weights[2] * gnn_output
        )
        
        return {
            'score': ensemble_output,
            'nn_score': nn_output,
            'attention_score': attended_output,
            'gnn_score': gnn_output,
            'attention_weights': attention_weights
        }

class CreditScoringEngine:
    """Main credit scoring engine"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rating_thresholds = {
            'AAA': 900, 'AA+': 850, 'AA': 800, 'AA-': 750,
            'A+': 700, 'A': 650, 'A-': 600,
            'BBB+': 550, 'BBB': 500, 'BBB-': 450,
            'BB+': 400, 'BB': 350, 'BB-': 300,
            'B+': 250, 'B': 200, 'B-': 150,
            'CCC': 100, 'CC': 50, 'C': 25, 'D': 0
        }
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Load pre-trained models if available
            self.ensemble_model = torch.load('models/ensemble_model.pth', map_location=self.device)
            self.xgb_model = joblib.load('models/xgb_model.pkl')
            self.rf_model = joblib.load('models/rf_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.feature_columns = joblib.load('models/feature_columns.pkl')
            self.is_trained = True
            logger.info("Loaded pre-trained models successfully")
        except FileNotFoundError:
            logger.info("No pre-trained models found, initializing new models")
            self._train_initial_models()
    
    def _train_initial_models(self):
        """Train initial models with synthetic data"""
        logger.info("Training initial models with synthetic data...")
        
        # Generate synthetic training data
        n_samples = 10000
        n_features = 50
        
        # Feature names
        self.feature_columns = [
            # Financial ratios
            'debt_to_equity', 'current_ratio', 'quick_ratio', 'debt_service_coverage',
            'interest_coverage', 'operating_margin', 'net_margin', 'roa', 'roe',
            'asset_turnover', 'inventory_turnover', 'receivables_turnover',
            
            # Market indicators
            'stock_volatility', 'market_cap', 'book_to_market', 'price_to_earnings',
            'price_to_book', 'dividend_yield', 'beta', 'sharpe_ratio',
            
            # Economic factors
            'gdp_growth', 'inflation_rate', 'unemployment_rate', 'interest_rates',
            'currency_strength', 'commodity_prices', 'vix_index', 'credit_spread',
            
            # Alternative data
            'news_sentiment', 'social_sentiment', 'satellite_activity', 'search_volume',
            'supply_chain_risk', 'cyber_risk_score', 'esg_score', 'management_quality',
            
            # Sector-specific
            'sector_performance', 'peer_comparison', 'regulatory_risk', 'competitive_position',
            'innovation_score', 'customer_satisfaction', 'employee_satisfaction', 'brand_strength',
            
            # Technical indicators
            'momentum_1m', 'momentum_3m', 'momentum_6m', 'volatility_30d', 'volume_trend', 'liquidity_score'
        ]
        
        # Generate synthetic features
        np.random.seed(42)
        X = np.random.randn(n_samples, len(self.feature_columns))
        
        # Create realistic relationships for credit scoring
        financial_health = (
            X[:, 0] * -0.3 +  # debt_to_equity (negative impact)
            X[:, 1] * 0.2 +   # current_ratio (positive impact)
            X[:, 7] * 0.25 +  # roa (positive impact)
            X[:, 8] * 0.2     # roe (positive impact)
        )
        
        market_sentiment = (
            X[:, 28] * 0.15 +  # news_sentiment
            X[:, 29] * 0.1 +   # social_sentiment
            X[:, 13] * -0.1    # stock_volatility (negative impact)
        )
        
        economic_impact = (
            X[:, 19] * 0.1 +   # gdp_growth
            X[:, 20] * -0.15 + # inflation_rate (negative impact)
            X[:, 22] * -0.1    # interest_rates (negative impact)
        )
        
        # Combine factors with noise
        y_continuous = (
            financial_health * 0.5 +
            market_sentiment * 0.3 +
            economic_impact * 0.2 +
            np.random.normal(0, 0.1, n_samples)  # noise
        )
        
        # Convert to credit scores (0-1000)
        y_scores = np.clip((y_continuous - y_continuous.min()) / 
                          (y_continuous.max() - y_continuous.min()) * 1000, 0, 1000)
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=self.feature_columns)
        df['credit_score'] = y_scores
        
        # Train models
        self._train_models(df)
        
    def _train_models(self, df: pd.DataFrame):
        """Train all models"""
        X = df[self.feature_columns].values
        y = df['credit_score'].values / 1000.0  # normalize to [0,1]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_model.fit(X_scaled, y)
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.rf_model.fit(X_scaled, y)
        
        # Train ensemble neural network
        self.ensemble_model = CreditScoringEnsemble(len(self.feature_columns))
        self.ensemble_model.to(self.device)
        
        # Training loop for neural network
        optimizer = torch.optim.Adam(self.ensemble_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        self.ensemble_model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.ensemble_model(X_tensor)
            loss = criterion(outputs['score'].squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        
        # Save models
        self._save_models()
        
    def _save_models(self):
        """Save trained models"""
        import os
        os.makedirs('models', exist_ok=True)
        
        torch.save(self.ensemble_model, 'models/ensemble_model.pth')
        joblib.dump(self.xgb_model, 'models/xgb_model.pkl')
        joblib.dump(self.rf_model, 'models/rf_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.feature_columns, 'models/feature_columns.pkl')
        
        logger.info("Models saved successfully")
    
    async def calculate_score(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate credit score for an entity"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        try:
            # Prepare features
            features = self._prepare_features(entity_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get predictions from all models
            xgb_score = self.xgb_model.predict(features_scaled)[0]
            rf_score = self.rf_model.predict(features_scaled)[0]
            
            # Neural network prediction
            self.ensemble_model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)
                nn_outputs = self.ensemble_model(features_tensor)
                nn_score = nn_outputs['score'].cpu().numpy()[0][0]
                attention_weights = nn_outputs['attention_weights'].cpu().numpy()
            
            # Ensemble prediction (weighted average)
            ensemble_score = (xgb_score * 0.4 + rf_score * 0.3 + nn_score * 0.3)
            
            # Convert to 0-1000 scale
            final_score = float(ensemble_score * 1000)
            final_score = max(0, min(1000, final_score))  # Clip to valid range
            
            # Determine rating
            rating = self._score_to_rating(final_score)
            
            # Calculate confidence based on model agreement
            scores = [xgb_score * 1000, rf_score * 1000, nn_score * 1000]
            confidence = 1.0 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0.5
            confidence = max(0.1, min(1.0, confidence))
            
            # Feature importance from XGBoost
            feature_importance = dict(zip(
                self.feature_columns,
                self.xgb_model.feature_importances_
            ))
            
            return {
                'score': final_score,
                'rating': rating,
                'confidence': confidence,
                'model_scores': {
                    'xgboost': float(xgb_score * 1000),
                    'random_forest': float(rf_score * 1000),
                    'neural_network': float(nn_score * 1000)
                },
                'feature_importance': feature_importance,
                'attention_weights': dict(zip(self.feature_columns, attention_weights.tolist())),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating credit score: {str(e)}")
            raise
    
    def _prepare_features(self, entity_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features from entity data"""
        features = np.zeros(len(self.feature_columns))
        
        for i, feature_name in enumerate(self.feature_columns):
            if feature_name in entity_data:
                features[i] = float(entity_data[feature_name])
            else:
                # Use default values or imputation
                features[i] = self._get_default_feature_value(feature_name)
        
        return features
    
    def _get_default_feature_value(self, feature_name: str) -> float:
        """Get default value for missing features"""
        defaults = {
            'debt_to_equity': 1.0,
            'current_ratio': 1.2,
            'quick_ratio': 1.0,
            'operating_margin': 0.1,
            'roa': 0.05,
            'roe': 0.1,
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'gdp_growth': 0.02,
            'inflation_rate': 0.02,
            'interest_rates': 0.03,
            'esg_score': 0.5,
            'beta': 1.0
        }
        return defaults.get(feature_name, 0.0)
    
    def _score_to_rating(self, score: float) -> str:
        """Convert numerical score to rating"""
        for rating, threshold in self.rating_thresholds.items():
            if score >= threshold:
                return rating
        return 'D'
    
    async def get_trends(self, entity_id: str, days: int = 30) -> Dict[str, Any]:
        """Get score trends for an entity"""
        # Simulate trend data (in real implementation, would query database)
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate realistic trend data
        base_score = np.random.uniform(400, 800)
        trend = np.random.normal(0, 10, len(dates)).cumsum()
        scores = np.clip(base_score + trend, 0, 1000)
        
        trend_data = []
        for date, score in zip(dates, scores):
            trend_data.append({
                'date': date.isoformat(),
                'score': float(score),
                'rating': self._score_to_rating(score)
            })
        
        # Calculate trend statistics
        recent_scores = scores[-7:]  # Last 7 days
        previous_scores = scores[-14:-7]  # Previous 7 days
        
        trend_direction = 'stable'
        if len(recent_scores) > 0 and len(previous_scores) > 0:
            recent_avg = np.mean(recent_scores)
            previous_avg = np.mean(previous_scores)
            change_pct = (recent_avg - previous_avg) / previous_avg * 100
            
            if change_pct > 2:
                trend_direction = 'improving'
            elif change_pct < -2:
                trend_direction = 'deteriorating'
        
        return {
            'entity_id': entity_id,
            'trend_data': trend_data,
            'trend_direction': trend_direction,
            'volatility': float(np.std(scores)),
            'current_score': float(scores[-1]),
            'change_30d': float(scores[-1] - scores[0]),
            'change_7d': float(scores[-1] - scores[-7]) if len(scores) >= 7 else 0.0
        }
    
    async def get_alerts(self, entity_id: str) -> List[str]:
        """Get alerts for an entity"""
        alerts = []
        
        # Simulate alert generation
        alert_conditions = [
            ("Score dropped more than 50 points in 24 hours", 0.1),
            ("Debt-to-equity ratio exceeds industry average", 0.15),
            ("Negative news sentiment spike detected", 0.2),
            ("Credit spread widening significantly", 0.1),
            ("Regulatory changes affecting sector", 0.05)
        ]
        
        for alert_text, probability in alert_conditions:
            if np.random.random() < probability:
                alerts.append(alert_text)
        
        return alerts
    
    async def get_historical_scores(
        self,
        entity_id: str,
        days: int,
        granularity: str = "daily"
    ) -> List[Dict[str, Any]]:
        """Get historical score data"""
        if granularity == "daily":
            freq = 'D'
        elif granularity == "hourly":
            freq = 'H'
        else:
            freq = 'D'
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq=freq
        )
        
        # Generate historical data with realistic patterns
        base_score = np.random.uniform(300, 900)
        volatility = np.random.uniform(5, 25)
        
        scores = []
        current_score = base_score
        
        for i, date in enumerate(dates):
            # Add some trend and noise
            trend = np.sin(i / len(dates) * 2 * np.pi) * 50
            noise = np.random.normal(0, volatility)
            current_score = np.clip(base_score + trend + noise, 0, 1000)
            
            scores.append({
                'timestamp': date.isoformat(),
                'score': float(current_score),
                'rating': self._score_to_rating(current_score)
            })
        
        return scores
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get market-wide risk overview"""
        # Simulate market data
        sectors = [
            'Technology', 'Healthcare', 'Financial', 'Energy',
            'Consumer', 'Industrial', 'Materials', 'Utilities'
        ]
        
        sector_data = []
        for sector in sectors:
            avg_score = np.random.uniform(400, 800)
            sector_data.append({
                'sector': sector,
                'average_score': float(avg_score),
                'average_rating': self._score_to_rating(avg_score),
                'trend': np.random.choice(['up', 'down', 'stable']),
                'entity_count': np.random.randint(10, 100)
            })
        
        market_sentiment = np.random.choice(['positive', 'negative', 'neutral'])
        volatility_index = np.random.uniform(10, 40)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'market_sentiment': market_sentiment,
            'volatility_index': float(volatility_index),
            'sector_overview': sector_data,
            'total_entities_tracked': sum([s['entity_count'] for s in sector_data]),
            'average_market_score': float(np.mean([s['average_score'] for s in sector_data]))
        }
    
    async def get_sector_analysis(self, sector: str) -> Dict[str, Any]:
        """Get detailed sector analysis"""
        # Simulate sector-specific analysis
        entity_count = np.random.randint(20, 200)
        entities = []
        
        for i in range(min(entity_count, 20)):  # Return top 20 entities
            score = np.random.uniform(200, 900)
            entities.append({
                'entity_id': f"{sector.lower()}_{i+1}",
                'name': f"{sector} Company {i+1}",
                'score': float(score),
                'rating': self._score_to_rating(score),
                'change_24h': float(np.random.uniform(-50, 50))
            })
        
        # Sort by score
        entities.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'sector': sector,
            'entity_count': entity_count,
            'average_score': float(np.mean([e['score'] for e in entities])),
            'top_entities': entities[:10],
            'bottom_entities': entities[-10:] if len(entities) >= 10 else [],
            'risk_distribution': {
                'low_risk': len([e for e in entities if e['score'] > 700]),
                'medium_risk': len([e for e in entities if 400 <= e['score'] <= 700]),
                'high_risk': len([e for e in entities if e['score'] < 400])
            }
        }
    
    async def start_continuous_scoring(self):
        """Start continuous scoring updates"""
        logger.info("Starting continuous scoring engine...")
        
        async def scoring_loop():
            while True:
                try:
                    # Update scores for all tracked entities
                    # In real implementation, would process entities from queue
                    await asyncio.sleep(60)  # Update every minute
                    logger.info("Scoring cycle completed")
                except Exception as e:
                    logger.error(f"Error in scoring loop: {str(e)}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(scoring_loop())
    
    async def stop_scoring(self):
        """Stop scoring engine"""
        logger.info("Stopping scoring engine...")
        # Cleanup tasks would go here
