# backend/requirements.txt
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
python-multipart==0.0.6

# Database & Caching
sqlalchemy==2.0.23
asyncpg==0.29.0
psycopg2-binary==2.9.7
redis==5.0.1
clickhouse-driver==0.2.6

# Message Queue
kafka-python==2.0.2
aiokafka==0.8.11

# Machine Learning
torch==2.1.0
torchvision==0.16.0
xgboost==1.7.6
scikit-learn==1.3.2
lightgbm==4.1.0
shap==0.43.0
networkx==3.2.1
numpy==1.24.3
pandas==2.0.3
scipy==1.11.4

# NLP & Text Processing
spacy==3.7.2
textblob==0.17.1
nltk==3.8.1
transformers==4.35.2

# Data Sources & APIs
yfinance==0.2.22
feedparser==6.0.10
requests==2.31.0
aiohttp==3.9.0
beautifulsoup4==4.12.2

# Visualization & Monitoring
plotly==5.17.0
matplotlib==3.8.2
seaborn==0.13.0
prometheus-client==0.19.0

# Utilities
python-dotenv==1.0.0
python-dateutil==2.8.2
pytz==2023.3
loguru==0.7.2
rich==13.7.0
typer==0.9.0
pyjwt==2.8.0
passlib[bcrypt]==1.7.4
python-jose[cryptography]==3.3.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2
factory-boy==3.3.0

# Development
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.5.0

# Deployment
gunicorn==21.2.0
docker==6.1.3
kubernetes==28.1.0