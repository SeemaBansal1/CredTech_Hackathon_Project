# backend/data_pipeline/data_ingestion.py
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
import hashlib
import yfinance as yf
import feedparser
from textblob import TextBlob
import sqlite3
import os

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    url: str
    source_type: str  # 'api', 'rss', 'web_scrape'
    update_frequency: int  # in seconds
    last_updated: Optional[datetime] = None
    is_active: bool = True
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, Any]] = None

class DatabaseManager:
    """Manages SQLite database for storing processed data"""
    
    def __init__(self, db_path: str = "data/credscope.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    name TEXT,
                    sector TEXT,
                    country TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Financial data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS financial_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    data_date TIMESTAMP,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    data_source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entity_id) REFERENCES entities (entity_id)
                )
            """)
            
            # News data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT,
                    title TEXT NOT NULL,
                    content TEXT,
                    url TEXT,
                    published_date TIMESTAMP,
                    source TEXT,
                    sentiment_score REAL,
                    relevance_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Scores history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS score_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    rating TEXT,
                    confidence REAL,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entity_id) REFERENCES entities (entity_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_financial_entity_date ON financial_data(entity_id, data_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_entity_date ON news_data(entity_id, published_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_score_entity_date ON score_history(entity_id, created_at)")
            
            conn.commit()
            logger.info("Database initialized successfully")

class YahooFinanceClient:
    """Client for Yahoo Finance data"""
    
    def __init__(self):
        self.session = None
    
    async def get_company_data(self, symbol: str) -> Dict[str, Any]:
        """Get company financial data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get recent price data
            hist = ticker.history(period="1mo")
            
            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return {}
            
            # Calculate financial metrics
            current_price = hist['Close'].iloc[-1] if not hist.empty else 0
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
            volume_trend = hist['Volume'].tail(5).mean() / hist['Volume'].tail(20).mean() if len(hist) >= 20 else 1.0
            
            # Extract key metrics
            data = {
                'symbol': symbol,
                'market_cap': info.get('marketCap', 0),
                'current_price': float(current_price),
                'stock_volatility': float(volatility) if not np.isnan(volatility) else 0.0,
                'volume_trend': float(volume_trend) if not np.isnan(volume_trend) else 1.0,
                'beta': info.get('beta', 1.0),
                'price_to_earnings': info.get('forwardPE', info.get('trailingPE', 0)),
                'price_to_book': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 1.0),
                'quick_ratio': info.get('quickRatio', 1.0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'return_on_assets': info.get('returnOnAssets', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {str(e)}")
            return {}

class NewsAPIClient:
    """Client for news data collection"""
    
    def __init__(self):
        self.session = None
        self.rss_feeds = [
            'http://feeds.reuters.com/reuters/businessNews',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://rss.cnn.com/rss/money_news_economy.rss',
            'https://feeds.finance.yahoo.com/rss/2.0/headline'
        ]
    
    async def get_news_for_entity(self, entity_name: str, entity_id: str) -> List[Dict[str, Any]]:
        """Get news articles related to an entity"""
        all_articles = []
        
        # Get RSS feed articles
        for feed_url in self.rss_feeds:
            try:
                articles = await self._get_rss_articles(feed_url, entity_name)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error fetching RSS from {feed_url}: {str(e)}")
        
        # Filter and score articles
        relevant_articles = []
        for article in all_articles:
            relevance_score = self._calculate_relevance(article, entity_name)
            if relevance_score > 0.3:  # Only include relevant articles
                article['entity_id'] = entity_id
                article['relevance_score'] = relevance_score
                article['sentiment_score'] = self._analyze_sentiment(article.get('content', ''))
                relevant_articles.append(article)
        
        return relevant_articles[:20]  # Return top 20 most relevant articles
    
    async def _get_rss_articles(self, feed_url: str, entity_name: str) -> List[Dict[str, Any]]:
        """Get articles from RSS feed"""
        articles = []
        
        try:
            # Use asyncio to run feedparser in a thread
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, feed_url)
            
            for entry in feed.entries[:50]:  # Limit to recent articles
                article = {
                    'title': entry.get('title', ''),
                    'content': entry.get('summary', entry.get('description', '')),
                    'url': entry.get('link', ''),
                    'published_date': self._parse_date(entry.get('published', '')),
                    'source': feed.feed.get('title', 'Unknown'),
                }
                
                # Only include if published within last 7 days
                if article['published_date'] and article['published_date'] > datetime.utcnow() - timedelta(days=7):
                    articles.append(article)
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {str(e)}")
        
        return articles
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        try:
            import dateutil.parser
            return dateutil.parser.parse(date_str)
        except:
            return None
    
    def _calculate_relevance(self, article: Dict[str, Any], entity_name: str) -> float:
        """Calculate relevance score for an article"""
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        entity_lower = entity_name.lower()
        
        relevance_score = 0.0
        
        # Direct entity name mentions
        title_mentions = title.count(entity_lower)
        content_mentions = content.count(entity_lower)
        
        relevance_score += title_mentions * 0.5  # Title mentions are more important
        relevance_score += content_mentions * 0.1
        
        # Related financial keywords
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'debt', 'merger',
            'acquisition', 'ipo', 'dividend', 'bankruptcy', 'credit rating',
            'financial results', 'quarterly', 'annual report'
        ]
        
        for keyword in financial_keywords:
            if keyword in title:
                relevance_score += 0.2
            if keyword in content:
                relevance_score += 0.05
        
        return min(1.0, relevance_score)
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        if not text:
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity  # Returns -1 to 1
        except:
            return 0.0

class EconomicDataClient:
    """Client for economic data from various sources"""
    
    def __init__(self):
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.world_bank_base_url = "https://api.worldbank.org/v2"
    
    async def get_economic_indicators(self, country_code: str = "US") -> Dict[str, Any]:
        """Get key economic indicators"""
        indicators = {}
        
        # Key economic series from FRED (for US data)
        if country_code == "US":
            fred_series = {
                'gdp_growth': 'A191RL1Q225SBEA',
                'inflation_rate': 'CPIAUCSL',
                'unemployment_rate': 'UNRATE',
                'interest_rates': 'FEDFUNDS',
                'vix_index': 'VIXCLS',
                'credit_spread': 'BAA10YM'
            }
            
            for indicator, series_id in fred_series.items():
                try:
                    value = await self._get_fred_data(series_id)
                    if value is not None:
                        indicators[indicator] = value
                except Exception as e:
                    logger.error(f"Error fetching {indicator}: {str(e)}")
                    indicators[indicator] = 0.0
        
        # Add commodity prices and currency data
        try:
            commodity_data = await self._get_commodity_prices()
            indicators.update(commodity_data)
        except Exception as e:
            logger.error(f"Error fetching commodity data: {str(e)}")
        
        indicators['timestamp'] = datetime.utcnow().isoformat()
        return indicators
    
    async def _get_fred_data(self, series_id: str) -> Optional[float]:
        """Get data from FRED API"""
        # Simulated FRED data since we don't have API key
        # In production, would use actual FRED API
        mock_data = {
            'A191RL1Q225SBEA': 2.1,  # GDP growth
            'CPIAUCSL': 3.2,         # Inflation rate
            'UNRATE': 3.7,           # Unemployment rate
            'FEDFUNDS': 5.25,        # Federal funds rate
            'VIXCLS': 18.5,          # VIX
            'BAA10YM': 1.8           # Credit spread
        }
        
        return mock_data.get(series_id)
    
    async def _get_commodity_prices(self) -> Dict[str, float]:
        """Get commodity prices"""
        # Mock commodity data
        return {
            'oil_price': np.random.uniform(70, 90),
            'gold_price': np.random.uniform(1800, 2100),
            'copper_price': np.random.uniform(3.5, 4.5),
            'currency_strength': np.random.uniform(0.95, 1.05)
        }

class AlternativeDataClient:
    """Client for alternative data sources"""
    
    def __init__(self):
        self.satellite_providers = []
        self.social_media_apis = []
    
    async def get_satellite_activity(self, entity_id: str, location: str = None) -> Dict[str, Any]:
        """Get economic activity indicators from satellite data"""
        # Mock satellite data - in production would integrate with providers like:
        # Planet Labs, Maxar, or other satellite imagery providers
        
        activity_score = np.random.uniform(0.3, 1.0)
        construction_activity = np.random.uniform(0, 1)
        shipping_activity = np.random.uniform(0, 1)
        
        return {
            'entity_id': entity_id,
            'satellite_activity': activity_score,
            'construction_indicator': construction_activity,
            'shipping_indicator': shipping_activity,
            'data_quality': 'medium',  # high, medium, low
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_social_sentiment(self, entity_name: str) -> Dict[str, Any]:
        """Get social media sentiment data"""
        # Mock social sentiment data
        # In production would integrate with Twitter API, Reddit API, etc.
        
        sentiment_score = np.random.uniform(-0.5, 0.5)
        mention_volume = np.random.randint(10, 1000)
        
        return {
            'social_sentiment': sentiment_score,
            'mention_volume': mention_volume,
            'engagement_rate': np.random.uniform(0.01, 0.1),
            'sentiment_volatility': np.random.uniform(0.1, 0.5),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_supply_chain_data(self, entity_id: str, sector: str) -> Dict[str, Any]:
        """Get supply chain risk indicators"""
        # Mock supply chain data
        base_risk = 0.3
        
        # Adjust risk based on sector
        sector_risk_multipliers = {
            'technology': 1.2,
            'automotive': 1.5,
            'healthcare': 0.8,
            'energy': 1.1,
            'financial': 0.6
        }
        
        risk_multiplier = sector_risk_multipliers.get(sector.lower(), 1.0)
        supply_chain_risk = base_risk * risk_multiplier * np.random.uniform(0.5, 1.5)
        
        return {
            'entity_id': entity_id,
            'supply_chain_risk': min(1.0, supply_chain_risk),
            'logistics_disruption': np.random.uniform(0, 0.5),
            'supplier_concentration': np.random.uniform(0.2, 0.8),
            'geographic_risk': np.random.uniform(0, 0.6),
            'timestamp': datetime.utcnow().isoformat()
        }

class DataIngestionManager:
    """Main data ingestion manager"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.yahoo_client = YahooFinanceClient()
        self.news_client = NewsAPIClient()
        self.economic_client = EconomicDataClient()
        self.alt_data_client = AlternativeDataClient()
        
        self.data_sources = []
        self.is_running = False
        self.tracked_entities = {}
        
        # Initialize with some sample entities
        self._init_sample_entities()
    
    def _init_sample_entities(self):
        """Initialize with sample entities for demonstration"""
        sample_entities = [
            {'entity_id': 'AAPL', 'name': 'Apple Inc.', 'type': 'company', 'sector': 'Technology'},
            {'entity_id': 'MSFT', 'name': 'Microsoft Corporation', 'type': 'company', 'sector': 'Technology'},
            {'entity_id': 'JPM', 'name': 'JPMorgan Chase', 'type': 'company', 'sector': 'Financial'},
            {'entity_id': 'XOM', 'name': 'Exxon Mobil', 'type': 'company', 'sector': 'Energy'},
            {'entity_id': 'JNJ', 'name': 'Johnson & Johnson', 'type': 'company', 'sector': 'Healthcare'},
            {'entity_id': 'TSLA', 'name': 'Tesla Inc.', 'type': 'company', 'sector': 'Automotive'},
            {'entity_id': 'US_TREASURY', 'name': 'US Treasury', 'type': 'sovereign', 'sector': 'Government'},
            {'entity_id': 'GERMANY_BUND', 'name': 'German Government', 'type': 'sovereign', 'sector': 'Government'}
        ]
        
        # Add entities to database
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            for entity in sample_entities:
                cursor.execute("""
                    INSERT OR REPLACE INTO entities 
                    (entity_id, entity_type, name, sector, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (entity['entity_id'], entity['type'], entity['name'], entity['sector']))
                
                self.tracked_entities[entity['entity_id']] = entity
            
            conn.commit()
        
        logger.info(f"Initialized {len(sample_entities)} sample entities")
    
    async def get_tracked_entities(self) -> List[Dict[str, Any]]:
        """Get list of all tracked entities"""
        entities = []
        
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity_id, entity_type, name, sector, country
                FROM entities
                ORDER BY name
            """)
            
            for row in cursor.fetchall():
                entities.append({
                    'entity_id': row[0],
                    'entity_type': row[1],
                    'name': row[2],
                    'sector': row[3],
                    'country': row[4]
                })
        
        return entities
    
    async def get_entity_data(self, entity_id: str) -> Dict[str, Any]:
        """Get comprehensive data for an entity"""
        if entity_id not in self.tracked_entities:
            logger.warning(f"Entity {entity_id} not found in tracked entities")
            return {}
        
        entity_info = self.tracked_entities[entity_id]
        entity_data = {
            'entity_id': entity_id,
            'entity_type': entity_info['type'],
            'name': entity_info['name'],
            'sector': entity_info['sector'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Get financial data (for companies)
            if entity_info['type'] == 'company':
                financial_data = await self.yahoo_client.get_company_data(entity_id)
                entity_data.update(financial_data)
            
            # Get economic indicators
            economic_data = await self.economic_client.get_economic_indicators()
            entity_data.update(economic_data)
            
            # Get alternative data
            alt_data = await self.alt_data_client.get_satellite_activity(entity_id)
            entity_data.update(alt_data)
            
            social_data = await self.alt_data_client.get_social_sentiment(entity_info['name'])
            entity_data.update(social_data)
            
            supply_chain_data = await self.alt_data_client.get_supply_chain_data(
                entity_id, entity_info['sector']
            )
            entity_data.update(supply_chain_data)
            
            # Add some derived features
            entity_data.update(self._calculate_derived_features(entity_data))
            
            # Store in database
            await self._store_entity_data(entity_data)
            
            return entity_data
            
        except Exception as e:
            logger.error(f"Error getting data for entity {entity_id}: {str(e)}")
            return entity_data
    
    def _calculate_derived_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived features from raw data"""
        derived = {}
        
        # Financial health score
        if data.get('current_ratio', 0) > 0 and data.get('debt_to_equity', 0) > 0:
            financial_health = (
                min(2.0, data.get('current_ratio', 1.0)) / 2.0 * 0.3 +
                max(0, 1 - data.get('debt_to_equity', 1.0) / 2.0) * 0.3 +
                max(0, data.get('operating_margin', 0.1)) * 5.0 * 0.4
            )
            derived['financial_health_score'] = min(1.0, financial_health)
        
        # Market sentiment composite
        news_sent = data.get('news_sentiment', 0.0) if 'news_sentiment' in data else 0.0
        social_sent = data.get('social_sentiment', 0.0)
        market_sentiment = (news_sent * 0.6 + social_sent * 0.4)
        derived['composite_sentiment'] = market_sentiment
        
        # Risk composite score
        supply_risk = data.get('supply_chain_risk', 0.3)
        market_vol = min(1.0, data.get('stock_volatility', 0.2) / 0.5)  # Normalize volatility
        risk_score = (supply_risk * 0.4 + market_vol * 0.6)
        derived['composite_risk_score'] = risk_score
        
        # Economic environment score
        gdp_score = max(0, min(1, (data.get('gdp_growth', 0.02) + 0.05) / 0.1))  # Normalize GDP growth
        inflation_penalty = max(0, (data.get('inflation_rate', 0.02) - 0.02) / 0.08)  # Penalty for high inflation
        econ_score = gdp_score - inflation_penalty
        derived['economic_environment_score'] = max(0, min(1, econ_score))
        
        return derived
    
    async def _store_entity_data(self, entity_data: Dict[str, Any]):
        """Store entity data in database"""
        entity_id = entity_data['entity_id']
        timestamp = datetime.utcnow()
        
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            # Store financial features
            financial_features = [
                'market_cap', 'current_price', 'stock_volatility', 'beta',
                'debt_to_equity', 'current_ratio', 'operating_margin',
                'return_on_equity', 'return_on_assets', 'gdp_growth',
                'inflation_rate', 'interest_rates', 'satellite_activity',
                'social_sentiment', 'supply_chain_risk', 'financial_health_score',
                'composite_sentiment', 'composite_risk_score'
            ]
            
            for feature in financial_features:
                if feature in entity_data and entity_data[feature] is not None:
                    cursor.execute("""
                        INSERT INTO financial_data 
                        (entity_id, data_date, feature_name, feature_value, data_source)
                        VALUES (?, ?, ?, ?, ?)
                    """, (entity_id, timestamp, feature, float(entity_data[feature]), 'ingestion_pipeline'))
            
            conn.commit()
    
    async def get_recent_news(self, entity_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent news for an entity"""
        if entity_id not in self.tracked_entities:
            return []
        
        entity_name = self.tracked_entities[entity_id]['name']
        
        try:
            news_articles = await self.news_client.get_news_for_entity(entity_name, entity_id)
            
            # Filter by time window
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_news = [
                article for article in news_articles
                if article.get('published_date') and article['published_date'] > cutoff_time
            ]
            
            # Store in database
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                for article in recent_news:
                    cursor.execute("""
                        INSERT OR REPLACE INTO news_data 
                        (entity_id, title, content, url, published_date, source, 
                         sentiment_score, relevance_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entity_id, article['title'], article.get('content', ''),
                        article.get('url', ''), article.get('published_date'),
                        article.get('source', ''), article.get('sentiment_score', 0.0),
                        article.get('relevance_score', 0.0)
                    ))
                
                conn.commit()
            
            return recent_news
            
        except Exception as e:
            logger.error(f"Error getting recent news for {entity_id}: {str(e)}")
            return []
    
    async def start_continuous_ingestion(self):
        """Start continuous data ingestion"""
        self.is_running = True
        logger.info("Starting continuous data ingestion...")
        
        async def ingestion_loop():
            while self.is_running:
                try:
                    # Update data for all tracked entities
                    for entity_id in self.tracked_entities.keys():
                        try:
                            await self.get_entity_data(entity_id)
                            logger.info(f"Updated data for {entity_id}")
                        except Exception as e:
                            logger.error(f"Error updating {entity_id}: {str(e)}")
                        
                        # Small delay between entities to avoid rate limiting
                        await asyncio.sleep(2)
                    
                    # Wait before next full cycle (5 minutes)
                    await asyncio.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Error in ingestion loop: {str(e)}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(ingestion_loop())
    
    async def stop_ingestion(self):
        """Stop data ingestion"""
        self.is_running = False
        logger.info("Stopping data ingestion...")
    
    async def add_entity(self, entity_id: str, name: str, entity_type: str, sector: str = None):
        """Add a new entity to track"""
        entity_info = {
            'entity_id': entity_id,
            'name': name,
            'type': entity_type,
            'sector': sector or 'Unknown'
        }
        
        # Add to database
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO entities 
                (entity_id, entity_type, name, sector, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (entity_id, entity_type, name, sector))
            conn.commit()
        
        # Add to tracked entities
        self.tracked_entities[entity_id] = entity_info
        
        logger.info(f"Added new entity: {entity_id} - {name}")
        
        # Get initial data
        await self.get_entity_data(entity_id)
    
    async def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality and freshness metrics"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            # Data freshness
            cursor.execute("""
                SELECT entity_id, MAX(created_at) as last_update
                FROM financial_data
                GROUP BY entity_id
            """)
            
            entity_freshness = {}
            for row in cursor.fetchall():
                entity_id, last_update = row
                last_update_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                hours_old = (datetime.utcnow() - last_update_dt.replace(tzinfo=None)).total_seconds() / 3600
                entity_freshness[entity_id] = hours_old
            
            # Data completeness
            cursor.execute("""
                SELECT COUNT(DISTINCT entity_id) as entities_with_data,
                       COUNT(DISTINCT feature_name) as total_features,
                       COUNT(*) as total_data_points
                FROM financial_data
                WHERE created_at > datetime('now', '-24 hours')
            """)
            
            completeness = cursor.fetchone()
            
            # News coverage
            cursor.execute("""
                SELECT COUNT(*) as news_articles,
                       COUNT(DISTINCT entity_id) as entities_with_news
                FROM news_data
                WHERE created_at > datetime('now', '-24 hours')
            """)
            
            news_stats = cursor.fetchone()
        
        return {
            'data_freshness': entity_freshness,
            'entities_with_fresh_data': len([h for h in entity_freshness.values() if h < 6]),  # < 6 hours old
            'total_features_collected': completeness[1] if completeness else 0,
            'total_data_points_24h': completeness[2] if completeness else 0,
            'news_articles_24h': news_stats[0] if news_stats else 0,
            'entities_with_news': news_stats[1] if news_stats else 0,
            'timestamp': datetime.utcnow().isoformat()
        }
