# backend/explainability/causal_engine.py
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import json
import shap
from scipy.stats import pearsonr
import re

logger = logging.getLogger(__name__)

class CausalGraph:
    """Builds and manages causal relationships between features"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.feature_relationships = {}
        self._build_domain_knowledge_graph()
    
    def _build_domain_knowledge_graph(self):
        """Build causal graph based on financial domain knowledge"""
        # Financial ratios relationships
        financial_causes = [
            ('debt_to_equity', 'interest_coverage', -0.7),
            ('current_ratio', 'debt_service_coverage', 0.6),
            ('operating_margin', 'roa', 0.8),
            ('roa', 'roe', 0.9),
            ('asset_turnover', 'roa', 0.7),
            
            # Market factors
            ('stock_volatility', 'beta', 0.8),
            ('market_cap', 'liquidity_score', 0.6),
            ('news_sentiment', 'stock_volatility', -0.5),
            ('social_sentiment', 'news_sentiment', 0.7),
            
            # Economic impacts
            ('gdp_growth', 'sector_performance', 0.6),
            ('inflation_rate', 'interest_rates', 0.8),
            ('interest_rates', 'debt_service_coverage', -0.7),
            ('unemployment_rate', 'consumer_spending', -0.8),
            
            # Alternative data
            ('satellite_activity', 'revenue_growth', 0.5),
            ('supply_chain_risk', 'operating_margin', -0.6),
            ('esg_score', 'brand_strength', 0.4),
            ('cyber_risk_score', 'operational_risk', 0.7)
        ]
        
        for cause, effect, strength in financial_causes:
            self.graph.add_edge(cause, effect, weight=strength)
            
        # Add nodes that don't have edges
        all_features = [
            'debt_to_equity', 'current_ratio', 'quick_ratio', 'debt_service_coverage',
            'interest_coverage', 'operating_margin', 'net_margin', 'roa', 'roe',
            'asset_turnover', 'stock_volatility', 'market_cap', 'beta', 'news_sentiment',
            'social_sentiment', 'gdp_growth', 'inflation_rate', 'interest_rates',
            'satellite_activity', 'supply_chain_risk', 'esg_score', 'cyber_risk_score',
            'sector_performance', 'liquidity_score', 'brand_strength'
        ]
        
        for feature in all_features:
            if feature not in self.graph.nodes():
                self.graph.add_node(feature)
    
    def get_causal_path(self, cause: str, effect: str) -> List[str]:
        """Find causal path between two features"""
        try:
            path = nx.shortest_path(self.graph, cause, effect)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def get_direct_causes(self, feature: str) -> List[Tuple[str, float]]:
        """Get direct causes of a feature"""
        causes = []
        for predecessor in self.graph.predecessors(feature):
            weight = self.graph[predecessor][feature]['weight']
            causes.append((predecessor, weight))
        return causes
    
    def get_direct_effects(self, feature: str) -> List[Tuple[str, float]]:
        """Get direct effects of a feature"""
        effects = []
        for successor in self.graph.successors(feature):
            weight = self.graph[feature][successor]['weight']
            effects.append((successor, weight))
        return effects

class EventImpactAnalyzer:
    """Analyzes impact of real-world events on credit scores"""
    
    def __init__(self):
        self.event_patterns = self._load_event_patterns()
        self.sentiment_keywords = self._load_sentiment_keywords()
    
    def _load_event_patterns(self) -> Dict[str, Dict]:
        """Load patterns for different types of events"""
        return {
            'debt_restructuring': {
                'keywords': ['debt restructuring', 'refinancing', 'debt relief', 'bankruptcy'],
                'impact_factors': {'debt_to_equity': 0.3, 'interest_coverage': -0.4},
                'severity_multiplier': 1.5,
                'time_decay': 0.1  # Impact decays over time
            },
            'earnings_warning': {
                'keywords': ['earnings warning', 'revenue decline', 'profit warning', 'guidance cut'],
                'impact_factors': {'operating_margin': -0.3, 'roa': -0.2, 'stock_volatility': 0.4},
                'severity_multiplier': 1.2,
                'time_decay': 0.15
            },
            'regulatory_action': {
                'keywords': ['regulatory', 'fine', 'investigation', 'compliance'],
                'impact_factors': {'regulatory_risk': 0.5, 'brand_strength': -0.3},
                'severity_multiplier': 1.3,
                'time_decay': 0.05
            },
            'acquisition_merger': {
                'keywords': ['acquisition', 'merger', 'takeover', 'buyout'],
                'impact_factors': {'debt_to_equity': 0.2, 'market_cap': 0.1},
                'severity_multiplier': 1.1,
                'time_decay': 0.2
            },
            'management_change': {
                'keywords': ['ceo change', 'management shake', 'leadership', 'resignation'],
                'impact_factors': {'management_quality': -0.2, 'stock_volatility': 0.2},
                'severity_multiplier': 1.0,
                'time_decay': 0.3
            }
        }
    
    def _load_sentiment_keywords(self) -> Dict[str, List[str]]:
        """Load sentiment analysis keywords"""
        return {
            'positive': [
                'strong', 'growth', 'profit', 'success', 'expansion', 'recovery',
                'improved', 'excellent', 'outstanding', 'beat', 'exceed', 'upgrade'
            ],
            'negative': [
                'weak', 'decline', 'loss', 'failure', 'contraction', 'recession',
                'poor', 'terrible', 'miss', 'below', 'downgrade', 'concern',
                'risk', 'warning', 'problem', 'issue', 'crisis'
            ]
        }
    
    def analyze_news_sentiment(self, news_text: str) -> Dict[str, Any]:
        """Analyze sentiment and extract key information from news"""
        if not news_text:
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'key_events': []}
        
        news_lower = news_text.lower()
        
        # Calculate sentiment score
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in news_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in news_lower)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            confidence = min(1.0, total_sentiment_words / 10.0)  # Higher confidence with more sentiment words
        else:
            sentiment_score = 0.0
            confidence = 0.0
        
        # Detect events
        detected_events = []
        for event_type, event_data in self.event_patterns.items():
            for keyword in event_data['keywords']:
                if keyword in news_lower:
                    detected_events.append({
                        'type': event_type,
                        'keyword': keyword,
                        'impact_factors': event_data['impact_factors'],
                        'severity': event_data['severity_multiplier']
                    })
                    break
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'positive_signals': positive_count,
            'negative_signals': negative_count,
            'detected_events': detected_events
        }

class CausalExplanationEngine:
    """Main engine for generating causal explanations"""
    
    def __init__(self):
        self.causal_graph = CausalGraph()
        self.event_analyzer = EventImpactAnalyzer()
        self.explanation_templates = self._load_explanation_templates()
        
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load templates for generating explanations"""
        return {
            'score_decrease': "The credit score decreased by {change:.1f} points primarily due to {main_factor}.",
            'score_increase': "The credit score improved by {change:.1f} points mainly because of {main_factor}.",
            'stable_score': "The credit score remained stable at {score:.1f} with minor fluctuations.",
            'causal_explanation': "This change was caused by {cause} which directly impacts {effect} through {mechanism}.",
            'event_impact': "Recent event: {event_type} - {event_description}. This typically affects {affected_factors}.",
            'trend_explanation': "The {trend_direction} trend over the past {period} indicates {trend_meaning}."
        }
    
    async def generate_explanation(
        self,
        entity_data: Dict[str, Any],
        score_result: Dict[str, Any],
        historical_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation for credit score"""
        
        try:
            explanation = {
                'timestamp': datetime.utcnow().isoformat(),
                'score_summary': self._generate_score_summary(score_result),
                'feature_analysis': self._analyze_feature_contributions(entity_data, score_result),
                'causal_relationships': self._explain_causal_relationships(entity_data, score_result),
                'risk_factors': self._identify_risk_factors(entity_data),
                'plain_language_summary': '',
                'recommendations': self._generate_recommendations(entity_data, score_result)
            }
            
            # Add historical context if available
            if historical_context:
                explanation['trend_analysis'] = self._analyze_trends(historical_context)
            
            # Generate plain language summary
            explanation['plain_language_summary'] = self._generate_plain_language_summary(explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {
                'error': 'Unable to generate explanation',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _generate_score_summary(self, score_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level score summary"""
        score = score_result['score']
        rating = score_result['rating']
        confidence = score_result['confidence']
        
        # Determine score category
        if score >= 700:
            category = 'low_risk'
            description = 'Low credit risk with strong fundamentals'
        elif score >= 500:
            category = 'medium_risk'
            description = 'Moderate credit risk requiring monitoring'
        else:
            category = 'high_risk'
            description = 'High credit risk with significant concerns'
        
        return {
            'score': score,
            'rating': rating,
            'category': category,
            'description': description,
            'confidence': confidence,
            'model_consensus': self._analyze_model_consensus(score_result.get('model_scores', {}))
        }
    
    def _analyze_model_consensus(self, model_scores: Dict[str, float]) -> str:
        """Analyze agreement between different models"""
        if not model_scores:
            return 'insufficient_data'
        
        scores = list(model_scores.values())
        std_dev = np.std(scores)
        mean_score = np.mean(scores)
        
        cv = std_dev / mean_score if mean_score > 0 else 1.0
        
        if cv < 0.05:
            return 'strong_consensus'
        elif cv < 0.1:
            return 'moderate_consensus'
        else:
            return 'weak_consensus'
    
    def _analyze_feature_contributions(
        self,
        entity_data: Dict[str, Any],
        score_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze how different features contribute to the score"""
        
        feature_importance = score_result.get('feature_importance', {})
        attention_weights = score_result.get('attention_weights', {})
        
        # Combine importance scores
        combined_importance = {}
        for feature in feature_importance.keys():
            importance = feature_importance.get(feature, 0.0)
            attention = attention_weights.get(feature, 0.0)
            combined_importance[feature] = (importance + attention) / 2.0
        
        # Sort by importance
        sorted_features = sorted(
            combined_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Categorize features
        top_positive = []
        top_negative = []
        
        for feature, importance in sorted_features[:20]:  # Top 20 features
            feature_value = entity_data.get(feature, 0.0)
            
            # Determine if feature is positive or negative contributor
            # This is simplified - in practice, would use SHAP values
            if feature_value > 0 and importance > 0:
                contribution_type = 'positive'
                top_positive.append({
                    'feature': feature,
                    'value': feature_value,
                    'importance': importance,
                    'human_readable': self._make_feature_human_readable(feature)
                })
            elif importance > 0.01:  # Only include significant negative contributors
                contribution_type = 'negative'
                top_negative.append({
                    'feature': feature,
                    'value': feature_value,
                    'importance': importance,
                    'human_readable': self._make_feature_human_readable(feature)
                })
        
        return {
            'top_positive_factors': top_positive[:5],
            'top_negative_factors': top_negative[:5],
            'feature_importance_distribution': dict(sorted_features[:10]),
            'total_features_analyzed': len(sorted_features)
        }
    
    def _make_feature_human_readable(self, feature_name: str) -> str:
        """Convert technical feature names to human-readable descriptions"""
        translations = {
            'debt_to_equity': 'Debt-to-Equity Ratio',
            'current_ratio': 'Current Ratio (Liquidity)',
            'quick_ratio': 'Quick Ratio (Liquid Assets)',
            'operating_margin': 'Operating Profit Margin',
            'roa': 'Return on Assets',
            'roe': 'Return on Equity',
            'stock_volatility': 'Stock Price Volatility',
            'news_sentiment': 'News Sentiment Score',
            'social_sentiment': 'Social Media Sentiment',
            'gdp_growth': 'Economic Growth Rate',
            'inflation_rate': 'Inflation Rate',
            'interest_rates': 'Interest Rate Environment',
            'satellite_activity': 'Economic Activity (Satellite Data)',
            'supply_chain_risk': 'Supply Chain Risk Score',
            'esg_score': 'Environmental, Social & Governance Score',
            'cyber_risk_score': 'Cybersecurity Risk Level'
        }
        
        return translations.get(feature_name, feature_name.replace('_', ' ').title())
    
    def _explain_causal_relationships(
        self,
        entity_data: Dict[str, Any],
        score_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Explain key causal relationships affecting the score"""
        
        relationships = []
        feature_importance = score_result.get('feature_importance', {})
        
        # Find top features and their causal relationships
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for feature, importance in top_features:
            if importance < 0.01:  # Skip low-importance features
                continue
                
            # Get causal relationships for this feature
            causes = self.causal_graph.get_direct_causes(feature)
            effects = self.causal_graph.get_direct_effects(feature)
            
            if causes or effects:
                relationship = {
                    'primary_feature': feature,
                    'human_readable': self._make_feature_human_readable(feature),
                    'importance': importance,
                    'current_value': entity_data.get(feature, 0.0),
                    'direct_causes': [
                        {
                            'cause': cause,
                            'human_readable': self._make_feature_human_readable(cause),
                            'strength': strength,
                            'current_value': entity_data.get(cause, 0.0)
                        }
                        for cause, strength in causes
                    ],
                    'direct_effects': [
                        {
                            'effect': effect,
                            'human_readable': self._make_feature_human_readable(effect),
                            'strength': strength
                        }
                        for effect, strength in effects
                    ],
                    'explanation': self._generate_causal_explanation(feature, causes, effects)
                }
                relationships.append(relationship)
        
        return relationships[:5]  # Return top 5 causal relationships
    
    def _generate_causal_explanation(
        self,
        feature: str,
        causes: List[Tuple[str, float]],
        effects: List[Tuple[str, float]]
    ) -> str:
        """Generate natural language explanation of causal relationships"""
        
        feature_readable = self._make_feature_human_readable(feature)
        
        explanation_parts = [f"{feature_readable} is a key factor in this credit assessment."]
        
        if causes:
            strong_causes = [cause for cause, strength in causes if abs(strength) > 0.5]
            if strong_causes:
                causes_readable = [self._make_feature_human_readable(c) for c in strong_causes[:2]]
                explanation_parts.append(
                    f"It is primarily influenced by {' and '.join(causes_readable)}."
                )
        
        if effects:
            strong_effects = [effect for effect, strength in effects if abs(strength) > 0.5]
            if strong_effects:
                effects_readable = [self._make_feature_human_readable(e) for e in strong_effects[:2]]
                explanation_parts.append(
                    f"Changes in {feature_readable} directly impact {' and '.join(effects_readable)}."
                )
        
        return " ".join(explanation_parts)
    
    def _identify_risk_factors(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify current and emerging risk factors"""
        
        risk_factors = {
            'current_risks': [],
            'emerging_risks': [],
            'risk_mitigation_opportunities': []
        }
        
        # Define risk thresholds
        risk_thresholds = {
            'debt_to_equity': {'high': 2.0, 'medium': 1.5},
            'current_ratio': {'low': 1.0, 'medium': 1.2},
            'operating_margin': {'low': 0.05, 'medium': 0.1},
            'stock_volatility': {'high': 0.3, 'medium': 0.2},
            'cyber_risk_score': {'high': 0.7, 'medium': 0.5},
            'supply_chain_risk': {'high': 0.6, 'medium': 0.4}
        }
        
        for feature, value in entity_data.items():
            if feature in risk_thresholds:
                thresholds = risk_thresholds[feature]
                
                if 'high' in thresholds and value >= thresholds['high']:
                    risk_factors['current_risks'].append({
                        'factor': feature,
                        'human_readable': self._make_feature_human_readable(feature),
                        'current_value': value,
                        'threshold': thresholds['high'],
                        'severity': 'high',
                        'description': f"High {self._make_feature_human_readable(feature).lower()} indicates significant risk"
                    })
                elif 'low' in thresholds and value <= thresholds['low']:
                    risk_factors['current_risks'].append({
                        'factor': feature,
                        'human_readable': self._make_feature_human_readable(feature),
                        'current_value': value,
                        'threshold': thresholds['low'],
                        'severity': 'high',
                        'description': f"Low {self._make_feature_human_readable(feature).lower()} indicates liquidity concerns"
                    })
                elif ('medium' in thresholds and 
                      ((('high' in thresholds and value >= thresholds['medium']) or
                        ('low' in thresholds and value <= thresholds['medium'])))):
                    risk_factors['emerging_risks'].append({
                        'factor': feature,
                        'human_readable': self._make_feature_human_readable(feature),
                        'current_value': value,
                        'threshold': thresholds['medium'],
                        'severity': 'medium',
                        'description': f"Moderate {self._make_feature_human_readable(feature).lower()} warrants monitoring"
                    })
        
        # Identify mitigation opportunities
        mitigation_opportunities = [
            {
                'opportunity': 'Debt Reduction',
                'description': 'Consider reducing debt levels to improve debt-to-equity ratio',
                'applicable_if': entity_data.get('debt_to_equity', 0) > 1.5,
                'potential_impact': 'high'
            },
            {
                'opportunity': 'Liquidity Improvement',
                'description': 'Increase liquid assets to improve current ratio',
                'applicable_if': entity_data.get('current_ratio', 0) < 1.2,
                'potential_impact': 'medium'
            },
            {
                'opportunity': 'Operational Efficiency',
                'description': 'Focus on improving operating margins through cost optimization',
                'applicable_if': entity_data.get('operating_margin', 0) < 0.1,
                'potential_impact': 'high'
            }
        ]
        
        risk_factors['risk_mitigation_opportunities'] = [
            opp for opp in mitigation_opportunities if opp['applicable_if']
        ]
        
        return risk_factors
    
    def _generate_recommendations(
        self,
        entity_data: Dict[str, Any],
        score_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        score = score_result['score']
        
        # Score-based recommendations
        if score < 400:
            recommendations.append({
                'priority': 'urgent',
                'category': 'financial_restructuring',
                'title': 'Immediate Financial Restructuring Required',
                'description': 'Consider comprehensive debt restructuring and liquidity management',
                'expected_impact': 'high',
                'timeframe': 'immediate'
            })
        elif score < 600:
            recommendations.append({
                'priority': 'high',
                'category': 'risk_management',
                'title': 'Strengthen Risk Management',
                'description': 'Implement enhanced risk monitoring and control measures',
                'expected_impact': 'medium',
                'timeframe': '3-6 months'
            })
        
        # Feature-specific recommendations
        feature_importance = score_result.get('feature_importance', {})
        top_negative_features = []
        
        for feature, importance in feature_importance.items():
            if importance > 0.05 and entity_data.get(feature, 0) < 0:  # Negative contributor
                top_negative_features.append((feature, importance))
        
        for feature, importance in sorted(top_negative_features, key=lambda x: x[1], reverse=True)[:3]:
            rec = self._get_feature_recommendation(feature, entity_data.get(feature, 0))
            if rec:
                recommendations.append(rec)
        
        # ESG and alternative data recommendations
        esg_score = entity_data.get('esg_score', 0.5)
        if esg_score < 0.4:
            recommendations.append({
                'priority': 'medium',
                'category': 'esg_improvement',
                'title': 'Improve ESG Performance',
                'description': 'Enhance environmental, social, and governance practices',
                'expected_impact': 'medium',
                'timeframe': '6-12 months'
            })
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _get_feature_recommendation(self, feature: str, current_value: float) -> Dict[str, Any]:
        """Get specific recommendation for a feature"""
        
        feature_recommendations = {
            'debt_to_equity': {
                'title': 'Reduce Debt Burden',
                'description': 'Lower debt-to-equity ratio through debt repayment or equity raising',
                'category': 'capital_structure'
            },
            'current_ratio': {
                'title': 'Improve Liquidity Position',
                'description': 'Increase current assets or reduce short-term liabilities',
                'category': 'liquidity_management'
            },
            'operating_margin': {
                'title': 'Enhance Operational Efficiency',
                'description': 'Improve operating margins through cost reduction and revenue optimization',
                'category': 'operational_improvement'
            },
            'news_sentiment': {
                'title': 'Improve Market Communication',
                'description': 'Enhance investor relations and public communication strategy',
                'category': 'stakeholder_management'
            }
        }
        
        if feature in feature_recommendations:
            base_rec = feature_recommendations[feature]
            return {
                'priority': 'medium',
                'category': base_rec['category'],
                'title': base_rec['title'],
                'description': base_rec['description'],
                'current_value': current_value,
                'expected_impact': 'medium',
                'timeframe': '3-6 months'
            }
        
        return None
    
    def _analyze_trends(self, historical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends from historical data"""
        
        trend_data = historical_context.get('trend_data', [])
        if not trend_data:
            return {'error': 'No historical data available'}
        
        scores = [item['score'] for item in trend_data]
        dates = [item['date'] for item in trend_data]
        
        if len(scores) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Calculate trend metrics
        recent_scores = scores[-7:] if len(scores) >= 7 else scores[-len(scores)//2:]
        older_scores = scores[:len(scores)//2]
        
        trend_direction = 'stable'
        if recent_scores and older_scores:
            recent_avg = np.mean(recent_scores)
            older_avg = np.mean(older_scores)
            change_pct = (recent_avg - older_avg) / older_avg * 100
            
            if change_pct > 5:
                trend_direction = 'improving'
            elif change_pct < -5:
                trend_direction = 'deteriorating'
        
        # Calculate volatility
        volatility = np.std(scores) if len(scores) > 1 else 0.0
        
        # Identify trend periods
        trend_periods = self._identify_trend_periods(scores, dates)
        
        return {
            'trend_direction': trend_direction,
            'volatility': float(volatility),
            'change_percentage': float(change_pct) if 'change_pct' in locals() else 0.0,
            'trend_periods': trend_periods,
            'current_vs_historical': {
                'current_score': scores[-1],
                'average_historical': float(np.mean(scores[:-1])),
                'percentile_rank': float(self._calculate_percentile_rank(scores[-1], scores[:-1]))
            }
        }
    
    def _identify_trend_periods(self, scores: List[float], dates: List[str]) -> List[Dict[str, Any]]:
        """Identify distinct trend periods in the data"""
        
        if len(scores) < 5:
            return []
        
        periods = []
        current_period = {'start_date': dates[0], 'start_score': scores[0]}
        current_trend = 'stable'
        
        for i in range(1, len(scores)):
            # Calculate local trend
            if i >= 3:  # Need at least 3 points to determine trend
                recent_trend = np.polyfit(range(3), scores[i-2:i+1], 1)[0]
                
                if recent_trend > 2:
                    new_trend = 'improving'
                elif recent_trend < -2:
                    new_trend = 'deteriorating'
                else:
                    new_trend = 'stable'
                
                # If trend changed, close current period and start new one
                if new_trend != current_trend:
                    current_period.update({
                        'end_date': dates[i-1],
                        'end_score': scores[i-1],
                        'trend': current_trend,
                        'duration_days': i - len(periods) * 10  # Approximate
                    })
                    periods.append(current_period)
                    
                    current_period = {
                        'start_date': dates[i-1],
                        'start_score': scores[i-1]
                    }
                    current_trend = new_trend
        
        # Close final period
        if current_period:
            current_period.update({
                'end_date': dates[-1],
                'end_score': scores[-1],
                'trend': current_trend,
                'duration_days': len(scores) - len(periods) * 10
            })
            periods.append(current_period)
        
        return periods[-5:]  # Return last 5 periods
    
    def _calculate_percentile_rank(self, value: float, historical_values: List[float]) -> float:
        """Calculate percentile rank of current value vs historical"""
        if not historical_values:
            return 50.0
        
        count_below = sum(1 for v in historical_values if v < value)
        return (count_below / len(historical_values)) * 100
    
    def _generate_plain_language_summary(self, explanation: Dict[str, Any]) -> str:
        """Generate human-readable summary of the explanation"""
        
        score_info = explanation['score_summary']
        score = score_info['score']
        rating = score_info['rating']
        category = score_info['category']
        
        summary_parts = []
        
        # Opening statement
        summary_parts.append(
            f"This entity has a credit score of {score:.0f} (rating: {rating}), "
            f"indicating {score_info['description'].lower()}."
        )
        
        # Key factors
        feature_analysis = explanation['feature_analysis']
        if feature_analysis['top_positive_factors']:
            top_positive = feature_analysis['top_positive_factors'][0]
            summary_parts.append(
                f"The strongest positive factor is {top_positive['human_readable'].lower()}, "
                f"which significantly supports the credit profile."
            )
        
        if feature_analysis['top_negative_factors']:
            top_negative = feature_analysis['top_negative_factors'][0]
            summary_parts.append(
                f"The main concern is {top_negative['human_readable'].lower()}, "
                f"which negatively impacts the overall assessment."
            )
        
        # Risk assessment
        risk_factors = explanation['risk_factors']
        if risk_factors['current_risks']:
            high_risks = [r for r in risk_factors['current_risks'] if r['severity'] == 'high']
            if high_risks:
                summary_parts.append(
                    f"There are {len(high_risks)} high-priority risk factors that require immediate attention."
                )
        
        # Recommendations
        recommendations = explanation['recommendations']
        if recommendations:
            urgent_recs = [r for r in recommendations if r['priority'] == 'urgent']
            if urgent_recs:
                summary_parts.append("Urgent action is required to address critical issues.")
            else:
                summary_parts.append("Several improvement opportunities have been identified.")
        
        # Model confidence
        confidence = score_info['confidence']
        if confidence > 0.8:
            summary_parts.append("The assessment has high confidence based on comprehensive data analysis.")
        elif confidence > 0.6:
            summary_parts.append("The assessment has moderate confidence with some data limitations.")
        else:
            summary_parts.append("The assessment has lower confidence due to limited or uncertain data.")
        
        return " ".join(summary_parts)
    
    async def analyze_news_impact(
        self,
        entity_id: str,
        news_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze impact of recent news on credit assessment"""
        
        if not news_data:
            return {'total_articles': 0, 'overall_impact': 'neutral'}
        
        impact_analysis = {
            'total_articles': len(news_data),
            'sentiment_analysis': {},
            'event_impacts': [],
            'overall_impact': 'neutral',
            'key_events': []
        }
        
        all_sentiment_scores = []
        all_events = []
        
        for article in news_data:
            title = article.get('title', '')
            content = article.get('content', '')
            published_date = article.get('published_date', '')
            
            # Analyze sentiment and events
            analysis = self.event_analyzer.analyze_news_sentiment(f"{title} {content}")
            
            if analysis['sentiment_score'] != 0.0:
                all_sentiment_scores.append(analysis['sentiment_score'])
            
            # Process detected events
            for event in analysis['detected_events']:
                event_with_context = {
                    **event,
                    'article_title': title,
                    'published_date': published_date,
                    'sentiment_score': analysis['sentiment_score']
                }
                all_events.append(event_with_context)
        
        # Overall sentiment analysis
        if all_sentiment_scores:
            avg_sentiment = np.mean(all_sentiment_scores)
            sentiment_std = np.std(all_sentiment_scores)
            
            impact_analysis['sentiment_analysis'] = {
                'average_sentiment': float(avg_sentiment),
                'sentiment_volatility': float(sentiment_std),
                'positive_articles': len([s for s in all_sentiment_scores if s > 0.1]),
                'negative_articles': len([s for s in all_sentiment_scores if s < -0.1]),
                'neutral_articles': len([s for s in all_sentiment_scores if -0.1 <= s <= 0.1])
            }
            
            # Determine overall impact
            if avg_sentiment > 0.2:
                impact_analysis['overall_impact'] = 'positive'
            elif avg_sentiment < -0.2:
                impact_analysis['overall_impact'] = 'negative'
        
        # Event impact analysis
        event_types = {}
        for event in all_events:
            event_type = event['type']
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        for event_type, events in event_types.items():
            impact_analysis['event_impacts'].append({
                'event_type': event_type,
                'frequency': len(events),
                'average_severity': np.mean([e['severity'] for e in events]),
                'affected_factors': list(set().union(*[e['impact_factors'].keys() for e in events])),
                'recent_examples': [
                    {
                        'title': e['article_title'],
                        'date': e['published_date'],
                        'sentiment': e['sentiment_score']
                    }
                    for e in events[:3]  # Show up to 3 examples
                ]
            })
        
        # Identify key events
        impact_analysis['key_events'] = sorted(
            all_events,
            key=lambda x: abs(x['sentiment_score']) * x['severity'],
            reverse=True
        )[:5]  # Top 5 most impactful events
        
        return impact_analysis
    
    async def analyze_scenario_impact(
        self,
        original_data: Dict[str, Any],
        modified_data: Dict[str, Any],
        original_score: Dict[str, Any],
        new_score: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the impact of a scenario change"""
        
        score_change = new_score['score'] - original_score['score']
        rating_change = new_score['rating'] != original_score['rating']
        
        # Identify changed features
        changed_features = []
        for feature in modified_data:
            if feature in original_data:
                original_value = original_data[feature]
                new_value = modified_data[feature]
                if abs(new_value - original_value) > 0.001:  # Significant change
                    change_pct = ((new_value - original_value) / original_value * 100) if original_value != 0 else 0
                    changed_features.append({
                        'feature': feature,
                        'human_readable': self._make_feature_human_readable(feature),
                        'original_value': original_value,
                        'new_value': new_value,
                        'change_percentage': change_pct
                    })
        
        # Analyze causal propagation
        causal_effects = []
        for changed_feature in changed_features:
            feature_name = changed_feature['feature']
            effects = self.causal_graph.get_direct_effects(feature_name)
            
            for effect_feature, strength in effects:
                causal_effects.append({
                    'cause': feature_name,
                    'effect': effect_feature,
                    'strength': strength,
                    'explanation': f"Change in {self._make_feature_human_readable(feature_name)} "
                                  f"affects {self._make_feature_human_readable(effect_feature)}"
                })
        
        return {
            'score_impact': {
                'original_score': original_score['score'],
                'new_score': new_score['score'],
                'score_change': score_change,
                'original_rating': original_score['rating'],
                'new_rating': new_score['rating'],
                'rating_changed': rating_change
            },
            'changed_features': changed_features,
            'causal_propagation': causal_effects,
            'impact_magnitude': abs(score_change),
            'impact_direction': 'positive' if score_change > 0 else 'negative' if score_change < 0 else 'neutral',
            'scenario_explanation': self._generate_scenario_explanation(
                changed_features, score_change, causal_effects
            )
        }
    
    def _generate_scenario_explanation(
        self,
        changed_features: List[Dict[str, Any]],
        score_change: float,
        causal_effects: List[Dict[str, Any]]
    ) -> str:
        """Generate explanation for scenario analysis"""
        
        if not changed_features:
            return "No significant changes were detected in the scenario."
        
        explanation_parts = []
        
        # Describe the changes
        if len(changed_features) == 1:
            feature = changed_features[0]
            explanation_parts.append(
                f"The scenario involves a {feature['change_percentage']:.1f}% change in "
                f"{feature['human_readable'].lower()}."
            )
        else:
            explanation_parts.append(
                f"The scenario involves changes to {len(changed_features)} factors: "
                f"{', '.join([f['human_readable'] for f in changed_features[:3]])}."
            )
        
        # Describe the impact
        if abs(score_change) > 20:
            impact_desc = "significant"
        elif abs(score_change) > 10:
            impact_desc = "moderate"
        else:
            impact_desc = "minor"
        
        direction = "positive" if score_change > 0 else "negative"
        explanation_parts.append(
            f"This results in a {impact_desc} {direction} impact on the credit score "
            f"({score_change:+.1f} points)."
        )
        
        # Describe causal propagation
        if causal_effects:
            key_effects = causal_effects[:2]  # Top 2 effects
            explanation_parts.append(
                f"The changes propagate through the system, affecting "
                f"{' and '.join([e['explanation'] for e in key_effects])}."
            )
        
        return " ".join(explanation_parts)
