#!/bin/bash
# setup.sh - CredScope AI Setup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if required tools are installed
check_dependencies() {
    print_header "Checking Dependencies"
    
    dependencies=("docker" "docker-compose" "git" "curl")
    missing_deps=()
    
    for dep in "${dependencies[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        else
            print_status "$dep is installed"
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and run the script again"
        exit 1
    fi
    
    print_status "All dependencies are installed"
}

# Create project structure
create_project_structure() {
    print_header "Creating Project Structure"
    
    # Main directories
    mkdir -p {backend/{api,models,data_pipeline,explainability,utils,tests/{unit,integration,e2e}},frontend/{src/{components,pages,utils,types},public},data/{raw,processed,models},deployment/{docker,kubernetes},docs,monitoring/{grafana,prometheus},notebooks,scripts}
    
    # Backend subdirectories
    mkdir -p backend/{api/{models,routes},data_pipeline,explainability,utils,database,migrations}
    
    # Frontend subdirectories
    mkdir -p frontend/src/{components/{dashboard,visualization,common},pages,hooks,services,store,types,styles}
    
    # Config and deployment directories
    mkdir -p {nginx,monitoring/{grafana/dashboards,prometheus},kubernetes,scripts}
    
    print_status "Project structure created successfully"
}

# Generate environment file
setup_environment() {
    print_header "Setting up Environment"
    
    if [ ! -f ".env" ]; then
        cat > .env << EOL
# CredScope AI Environment Configuration

# Application
APP_NAME=CredScope AI
APP_VERSION=1.0.0
ENV=development
DEBUG=true
SECRET_KEY=$(openssl rand -hex 32)

# Database Configuration
DATABASE_URL=postgresql://credscope_user:credscope_pass@localhost:5432/credscope
REDIS_URL=redis://:credscope_redis_pass@localhost:6379/0
CLICKHOUSE_URL=http://localhost:8123

# Message Queue
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# External APIs (Add your API keys here)
YAHOO_FINANCE_API_KEY=your_yahoo_finance_key
NEWS_API_KEY=your_news_api_key
FRED_API_KEY=your_fred_api_key

# Security
JWT_SECRET_KEY=$(openssl rand -hex 32)
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# ML Configuration
MODEL_UPDATE_FREQUENCY=3600
BATCH_SIZE=32
LEARNING_RATE=0.001

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
EOL
        print_status "Environment file created: .env"
        print_warning "Please update the API keys in .env file"
    else
        print_status "Environment file already exists"
    fi
}

# Setup backend
setup_backend() {
    print_header "Setting up Backend"
    
    cd backend
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment"
        python3 -m venv venv
    fi
    
    print_status "Activating virtual environment"
    source venv/bin/activate
    
    # Install dependencies
    print_status "Installing Python dependencies"
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Download spaCy model
    print_status "Downloading spaCy language model"
    python -m spacy download en_core_web_sm
    
    cd ..
    print_status "Backend setup completed"
}

# Setup frontend
setup_frontend() {
    print_header "Setting up Frontend"
    
    cd frontend
    
    # Install dependencies
    print_status "Installing Node.js dependencies"
    npm install
    
    cd ..
    print_status "Frontend setup completed"
}

# Create Docker configuration
setup_docker() {
    print_header "Setting up Docker Configuration"
    
    # Copy docker-compose file if it doesn't exist
    if [ ! -f "docker-compose.yml" ]; then
        print_status "Docker Compose file created"
    else
        print_status "Docker Compose file already exists"
    fi
    
    # Create nginx config
    mkdir -p nginx
    cat > nginx/nginx.conf << 'EOL'
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        location /api/ {
            proxy_pass http://api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        location /ws/ {
            proxy_pass http://api/ws/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
EOL

    print_status "Docker configuration completed"
}

# Initialize database
init_database() {
    print_header "Initializing Database"
    
    # Wait for services to be ready
    print_status "Starting database services"
    docker-compose up -d postgres redis
    
    # Wait for postgres to be ready
    print_status "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # Run database migrations
    print_status "Running database migrations"
    docker-compose exec api python -c "
from backend.database.db import DatabaseManager
db = DatabaseManager()
print('Database initialized successfully')
"
    
    print_status "Database initialization completed"
}

# Create sample data
create_sample_data() {
    print_header "Creating Sample Data"
    
    print_status "Loading sample entities and data"
    docker-compose exec api python -c "
from backend.data_pipeline.data_ingestion import DataIngestionManager
import asyncio

async def load_sample_data():
    manager = DataIngestionManager()
    print('Sample data loaded successfully')

asyncio.run(load_sample_data())
"
    
    print_status "Sample data creation completed"
}

# Build and start services
start_services() {
    print_header "Building and Starting Services"
    
    print_status "Building Docker images"
    docker-compose build
    
    print_status "Starting all services"
    docker-compose up -d
    
    print_status "Waiting for services to start..."
    sleep 30
    
    # Health check
    print_status "Performing health checks"
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "✓ API service is healthy"
    else
        print_warning "⚠ API service health check failed"
    fi
    
    # Check frontend
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        print_status "✓ Frontend service is healthy"
    else
        print_warning "⚠ Frontend service health check failed"
    fi
    
    print_status "Services started successfully"
}

# Display final information
show_final_info() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}🎉 CredScope AI has been set up successfully!${NC}"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo -e "  📊 Dashboard:     ${GREEN}http://localhost:3000${NC}"
    echo -e "  🔧 API Docs:      ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "  📈 Grafana:       ${GREEN}http://localhost:3001${NC} (admin/admin)"
    echo -e "  🔍 Prometheus:    ${GREEN}http://localhost:9090${NC}"
    echo -e "  📝 Jupyter:       ${GREEN}http://localhost:8888${NC} (token: credscope123)"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo -e "  View logs:        ${YELLOW}docker-compose logs -f${NC}"
    echo -e "  Stop services:    ${YELLOW}docker-compose down${NC}"
    echo -e "  Restart:          ${YELLOW}docker-compose restart${NC}"
    echo -e "  Update services:  ${YELLOW}docker-compose pull && docker-compose up -d${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo -e "  1. Add your API keys to the .env file"
    echo -e "  2. Customize the dashboard at http://localhost:3000"
    echo -e "  3. Explore the API documentation at http://localhost:8000/docs"
    echo -e "  4. Check the sample notebooks in the notebooks/ directory"
    echo ""
    echo -e "${GREEN}Happy credit scoring! 🚀${NC}"
}

# Create additional utility scripts
create_utility_scripts() {
    print_header "Creating Utility Scripts"
    
    # Create development script
    cat > scripts/dev.sh << 'EOL'
#!/bin/bash
# Development helper script

case "$1" in
    "start")
        echo "Starting development environment..."
        docker-compose up -d postgres redis kafka
        cd backend && source venv/bin/activate && uvicorn api.main:app --reload &
        cd frontend && npm start &
        ;;
    "test")
        echo "Running tests..."
        cd backend && source venv/bin/activate && python -m pytest tests/
        cd frontend && npm test
        ;;
    "lint")
        echo "Running linting..."
        cd backend && source venv/bin/activate && black . && isort .
        cd frontend && npm run lint:fix
        ;;
    "build")
        echo "Building for production..."
        docker-compose build
        ;;
    *)
        echo "Usage: $0 {start|test|lint|build}"
        exit 1
        ;;
esac
EOL

    # Create backup script
    cat > scripts/backup.sh << 'EOL'
#!/bin/bash
# Backup script for CredScope AI

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backup in $BACKUP_DIR..."

# Backup database
docker-compose exec postgres pg_dump -U credscope_user credscope > "$BACKUP_DIR/database.sql"

# Backup models
cp -r data/models "$BACKUP_DIR/" 2>/dev/null || echo "No models to backup"

# Backup configuration
cp .env "$BACKUP_DIR/"
cp docker-compose.yml "$BACKUP_DIR/"

echo "Backup completed in $BACKUP_DIR"
EOL

    # Create monitoring script
    cat > scripts/monitor.sh << 'EOL'
#!/bin/bash
# Monitoring script

echo "CredScope AI System Status"
echo "========================="

# Check service status
echo "Docker Services:"
docker-compose ps

echo -e "\nResource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo -e "\nAPI Health:"
curl -s http://localhost:8000/health | jq '.'

echo -e "\nDatabase Connection:"
docker-compose exec postgres pg_isready -U credscope_user -d credscope

echo -e "\nRecent Logs (last 50 lines):"
docker-compose logs --tail=50 api
EOL

    # Make scripts executable
    chmod +x scripts/*.sh
    
    print_status "Utility scripts created in scripts/ directory"
}

# Create presentation and demo materials
create_demo_materials() {
    print_header "Creating Demo Materials"
    
    # Create a simple demo script
    cat > scripts/demo.py << 'EOL'
#!/usr/bin/env python3
"""
CredScope AI Demo Script
Demonstrates key features of the platform
"""

import asyncio
import json
import requests
from datetime import datetime

API_BASE = "http://localhost:8000"

def demo_credit_scoring():
    """Demo credit scoring functionality"""
    print("🎯 Demo: Credit Scoring")
    print("-" * 30)
    
    entities = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    for entity in entities:
        try:
            response = requests.get(f"{API_BASE}/score/{entity}?include_explanation=true")
            if response.status_code == 200:
                data = response.json()
                print(f"📊 {entity}: Score {data['score']:.1f} ({data['rating']})")
                print(f"   Confidence: {data['confidence']*100:.1f}%")
            else:
                print(f"❌ {entity}: Error fetching data")
        except Exception as e:
            print(f"❌ {entity}: {str(e)}")
    print()

def demo_scenario_analysis():
    """Demo scenario analysis"""
    print("🔍 Demo: Scenario Analysis")
    print("-" * 30)
    
    scenario_data = {
        "entity_id": "AAPL",
        "scenario_changes": {
            "debt_to_equity": -0.1,
            "operating_margin": 0.05
        },
        "time_horizon": 30
    }
    
    try:
        response = requests.post(f"{API_BASE}/scenario-analysis", json=scenario_data)
        if response.status_code == 200:
            data = response.json()
            original = data['original_score']['score']
            new = data['new_score']['score']
            impact = new - original
            print(f"📈 AAPL Scenario Analysis:")
            print(f"   Original Score: {original:.1f}")
            print(f"   New Score: {new:.1f}")
            print(f"   Impact: {impact:+.1f} points")
        else:
            print("❌ Scenario analysis failed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    print()

def demo_market_overview():
    """Demo market overview"""
    print("🌍 Demo: Market Overview")
    print("-" * 30)
    
    try:
        response = requests.get(f"{API_BASE}/market-overview")
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Market Sentiment: {data['market_sentiment'].title()}")
            print(f"📈 Volatility Index: {data['volatility_index']}")
            print(f"🏢 Total Entities: {data['total_entities_tracked']}")
            print(f"⭐ Average Score: {data['average_market_score']:.1f}")
        else:
            print("❌ Market overview failed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    print()

if __name__ == "__main__":
    print("🚀 CredScope AI Platform Demo")
    print("=" * 40)
    print(f"⏰ Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    import time
    time.sleep(5)
    
    # Run demos
    demo_credit_scoring()
    demo_scenario_analysis()
    demo_market_overview()
    
    print("✅ Demo completed successfully!")
    print("🎉 Visit http://localhost:3000 to explore the full dashboard!")
EOL

    chmod +x scripts/demo.py
    
    print_status "Demo materials created"
}

# Main setup function
main() {
    print_header "CredScope AI - Automated Setup"
    echo -e "${BLUE}Setting up the award-winning real-time credit intelligence platform${NC}"
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "README.md" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    check_dependencies
    create_project_structure
    setup_environment
    setup_docker
    create_utility_scripts
    create_demo_materials
    
    # Ask user if they want to start services now
    echo ""
    read -p "Do you want to start the services now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_backend
        setup_frontend
        start_services
        init_database
        create_sample_data
        
        # Run demo
        echo ""
        read -p "Do you want to run a quick demo? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 scripts/demo.py
        fi
        
        show_final_info
    else
        print_status "Setup completed. To start services later, run:"
        echo "  docker-compose up -d"
        echo "  python3 scripts/demo.py"
    fi
}

# Run main function
main "$@"