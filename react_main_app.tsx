import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, TrendingDown, AlertTriangle, Eye, 
  BarChart3, Globe, Users, Settings, RefreshCw,
  Activity, DollarSign, Shield, Zap
} from 'lucide-react';

// Mock data for demonstration
const mockEntities = [
  { 
    entity_id: 'AAPL', 
    name: 'Apple Inc.', 
    score: 785, 
    rating: 'AA-', 
    change: 12, 
    sector: 'Technology',
    trend: 'up'
  },
  { 
    entity_id: 'MSFT', 
    name: 'Microsoft Corporation', 
    score: 820, 
    rating: 'AA+', 
    change: -5, 
    sector: 'Technology',
    trend: 'down'
  },
  { 
    entity_id: 'JPM', 
    name: 'JPMorgan Chase', 
    score: 740, 
    rating: 'A+', 
    change: 8, 
    sector: 'Financial',
    trend: 'up'
  },
  { 
    entity_id: 'XOM', 
    name: 'Exxon Mobil', 
    score: 680, 
    rating: 'A-', 
    change: -15, 
    sector: 'Energy',
    trend: 'down'
  },
  { 
    entity_id: 'TSLA', 
    name: 'Tesla Inc.', 
    score: 620, 
    rating: 'BBB+', 
    change: 25, 
    sector: 'Automotive',
    trend: 'up'
  },
];

const mockMarketData = {
  market_sentiment: 'positive',
  volatility_index: 18.5,
  total_entities: 128,
  average_score: 725
};

const Dashboard = () => {
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [entities, setEntities] = useState(mockEntities);
  const [marketData, setMarketData] = useState(mockMarketData);
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(false);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setEntities(prev => prev.map(entity => ({
        ...entity,
        score: entity.score + Math.random() * 10 - 5,
        change: entity.change + Math.random() * 4 - 2
      })));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const ScoreCard = ({ entity, onClick }) => {
    const getTrendColor = (trend, change) => {
      if (change > 0) return 'text-green-500';
      if (change < 0) return 'text-red-500';
      return 'text-gray-500';
    };

    const getRatingColor = (rating) => {
      if (rating.startsWith('AA')) return 'bg-green-100 text-green-800';
      if (rating.startsWith('A')) return 'bg-blue-100 text-blue-800';
      if (rating.startsWith('BBB')) return 'bg-yellow-100 text-yellow-800';
      if (rating.startsWith('BB') || rating.startsWith('B')) return 'bg-orange-100 text-orange-800';
      return 'bg-red-100 text-red-800';
    };

    return (
      <div 
        onClick={() => onClick(entity)}
        className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 p-6 cursor-pointer border hover:border-blue-300"
      >
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="font-bold text-lg text-gray-900">{entity.name}</h3>
            <p className="text-gray-500 text-sm">{entity.sector}</p>
          </div>
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${getRatingColor(entity.rating)}`}>
            {entity.rating}
          </span>
        </div>
        
        <div className="flex items-center justify-between">
          <div>
            <div className="text-3xl font-bold text-gray-900">{Math.round(entity.score)}</div>
            <div className="text-sm text-gray-500">Credit Score</div>
          </div>
          
          <div className={`flex items-center space-x-1 ${getTrendColor(entity.trend, entity.change)}`}>
            {entity.change > 0 ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
            <span className="font-medium">{entity.change > 0 ? '+' : ''}{Math.round(entity.change)}</span>
          </div>
        </div>
      </div>
    );
  };

  const EntityDetailView = ({ entity }) => {
    const [detailTab, setDetailTab] = useState('overview');
    
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{entity.name}</h2>
            <p className="text-gray-500">{entity.sector} • {entity.entity_id}</p>
          </div>
          <button
            onClick={() => setSelectedEntity(null)}
            className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
          >
            Back to Overview
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-blue-600">Credit Score</h3>
              <Activity className="text-blue-500" size={20} />
            </div>
            <div className="text-3xl font-bold text-blue-900">{Math.round(entity.score)}</div>
            <div className="text-sm text-blue-600">Rating: {entity.rating}</div>
          </div>
          
          <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-green-600">24h Change</h3>
              {entity.change > 0 ? <TrendingUp className="text-green-500" size={20} /> : <TrendingDown className="text-red-500" size={20} />}
            </div>
            <div className={`text-3xl font-bold ${entity.change > 0 ? 'text-green-900' : 'text-red-900'}`}>
              {entity.change > 0 ? '+' : ''}{Math.round(entity.change)}
            </div>
            <div className="text-sm text-green-600">Points</div>
          </div>
          
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-purple-600">Confidence</h3>
              <Shield className="text-purple-500" size={20} />
            </div>
            <div className="text-3xl font-bold text-purple-900">87%</div>
            <div className="text-sm text-purple-600">Model Consensus</div>
          </div>
        </div>

        <div className="border-b border-gray-200 mb-6">
          <nav className="flex space-x-8">
            {['overview', 'explanation', 'trends', 'news', 'scenarios'].map(tab => (
              <button
                key={tab}
                onClick={() => setDetailTab(tab)}
                className={`py-2 px-1 border-b-2 font-medium text-sm capitalize transition-colors ${
                  detailTab === tab
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>

        <div className="space-y-6">
          {detailTab === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900">Key Metrics</h3>
                <div className="space-y-3">
                  <MetricItem label="Debt-to-Equity Ratio" value="0.85" trend="good" />
                  <MetricItem label="Current Ratio" value="1.24" trend="good" />
                  <MetricItem label="Operating Margin" value="15.2%" trend="excellent" />
                  <MetricItem label="ROE" value="18.4%" trend="excellent" />
                  <MetricItem label="Stock Volatility" value="0.23" trend="moderate" />
                </div>
              </div>
              
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900">Risk Factors</h3>
                <div className="space-y-3">
                  <RiskItem 
                    risk="Supply Chain Risk" 
                    level="Medium" 
                    description="Moderate exposure to supply chain disruptions"
                  />
                  <RiskItem 
                    risk="Market Volatility" 
                    level="Low" 
                    description="Below-average market volatility impact"
                  />
                  <RiskItem 
                    risk="Regulatory Risk" 
                    level="Low" 
                    description="Limited regulatory exposure in current markets"
                  />
                </div>
              </div>
            </div>
          )}

          {detailTab === 'explanation' && (
            <ExplanationView entity={entity} />
          )}

          {detailTab === 'trends' && (
            <TrendsView entity={entity} />
          )}

          {detailTab === 'news' && (
            <NewsView entity={entity} />
          )}

          {detailTab === 'scenarios' && (
            <ScenarioView entity={entity} />
          )}
        </div>
      </div>
    );
  };

  const MetricItem = ({ label, value, trend }) => {
    const getTrendColor = (trend) => {
      if (trend === 'excellent') return 'text-green-600 bg-green-50';
      if (trend === 'good') return 'text-blue-600 bg-blue-50';
      if (trend === 'moderate') return 'text-yellow-600 bg-yellow-50';
      return 'text-red-600 bg-red-50';
    };

    return (
      <div className="flex justify-between items-center py-2">
        <span className="text-gray-700">{label}</span>
        <div className="flex items-center space-x-2">
          <span className="font-medium text-gray-900">{value}</span>
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTrendColor(trend)}`}>
            {trend}
          </span>
        </div>
      </div>
    );
  };

  const RiskItem = ({ risk, level, description }) => {
    const getLevelColor = (level) => {
      if (level === 'Low') return 'text-green-600 bg-green-50';
      if (level === 'Medium') return 'text-yellow-600 bg-yellow-50';
      return 'text-red-600 bg-red-50';
    };

    return (
      <div className="p-3 bg-gray-50 rounded-lg">
        <div className="flex justify-between items-center mb-1">
          <span className="font-medium text-gray-900">{risk}</span>
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getLevelColor(level)}`}>
            {level}
          </span>
        </div>
        <p className="text-sm text-gray-600">{description}</p>
      </div>
    );
  };

  const ExplanationView = ({ entity }) => (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">Score Explanation</h3>
        <p className="text-blue-800 leading-relaxed">
          {entity.name} has a credit score of {Math.round(entity.score)} (rating: {entity.rating}), 
          indicating strong credit fundamentals. The score is primarily supported by excellent 
          operational efficiency metrics and strong balance sheet indicators.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h4 className="font-semibold text-gray-900">Positive Factors</h4>
          <div className="space-y-3">
            <FactorItem 
              factor="Operating Margin" 
              impact="High" 
              description="Strong operational efficiency with 15.2% margin"
              positive={true}
            />
            <FactorItem 
              factor="Return on Equity" 
              impact="High" 
              description="Excellent ROE of 18.4% indicates effective use of shareholder equity"
              positive={true}
            />
            <FactorItem 
              factor="Current Ratio" 
              impact="Medium" 
              description="Healthy liquidity position with ratio of 1.24"
              positive={true}
            />
          </div>
        </div>

        <div className="space-y-4">
          <h4 className="font-semibold text-gray-900">Areas of Concern</h4>
          <div className="space-y-3">
            <FactorItem 
              factor="Market Volatility" 
              impact="Medium" 
              description="Recent increase in stock price volatility"
              positive={false}
            />
            <FactorItem 
              factor="Supply Chain Risk" 
              impact="Low" 
              description="Moderate exposure to global supply chain disruptions"
              positive={false}
            />
          </div>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h4 className="font-semibold text-gray-900 mb-3">Causal Relationships</h4>
        <div className="space-y-3">
          <CausalRelationship 
            cause="Strong Operating Margins"
            effect="Higher ROA and ROE"
            explanation="Efficient operations directly translate to better asset utilization and shareholder returns"
          />
          <CausalRelationship 
            cause="Stable Market Position"
            effect="Lower Credit Risk"
            explanation="Market leadership provides defensive characteristics against economic downturns"
          />
        </div>
      </div>
    </div>
  );

  const FactorItem = ({ factor, impact, description, positive }) => (
    <div className={`p-3 rounded-lg border ${positive ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
      <div className="flex justify-between items-center mb-1">
        <span className={`font-medium ${positive ? 'text-green-900' : 'text-red-900'}`}>{factor}</span>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
          impact === 'High' ? 'bg-gray-800 text-white' : 
          impact === 'Medium' ? 'bg-gray-600 text-white' : 'bg-gray-400 text-white'
        }`}>
          {impact} Impact
        </span>
      </div>
      <p className={`text-sm ${positive ? 'text-green-700' : 'text-red-700'}`}>{description}</p>
    </div>
  );

  const CausalRelationship = ({ cause, effect, explanation }) => (
    <div className="p-3 bg-white rounded border">
      <div className="flex items-center space-x-3 mb-2">
        <span className="text-sm font-medium text-blue-600">{cause}</span>
        <span className="text-gray-400">→</span>
        <span className="text-sm font-medium text-green-600">{effect}</span>
      </div>
      <p className="text-sm text-gray-600">{explanation}</p>
    </div>
  );

  const TrendsView = ({ entity }) => (
    <div className="space-y-6">
      <div className="bg-white border rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Score Trend (30 Days)</h3>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-center text-gray-500">
            <BarChart3 size={48} className="mx-auto mb-2" />
            <p>Interactive trend chart would be displayed here</p>
            <p className="text-sm">Showing 30-day score history with key events</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <h4 className="font-semibold text-blue-900 mb-2">7-Day Change</h4>
          <div className="text-2xl font-bold text-blue-900">+12 pts</div>
          <p className="text-sm text-blue-700">Improving trend</p>
        </div>
        
        <div className="bg-green-50 rounded-lg p-4">
          <h4 className="font-semibold text-green-900 mb-2">30-Day Volatility</h4>
          <div className="text-2xl font-bold text-green-900">18.5</div>
          <p className="text-sm text-green-700">Below average</p>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-4">
          <h4 className="font-semibold text-purple-900 mb-2">Percentile Rank</h4>
          <div className="text-2xl font-bold text-purple-900">78th</div>
          <p className="text-sm text-purple-700">vs. sector peers</p>
        </div>
      </div>
    </div>
  );

  const NewsView = ({ entity }) => (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900">Recent News Impact</h3>
        <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          Refresh News
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <div className="bg-green-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-900">+2.5</div>
          <div className="text-sm text-green-700">Positive News Impact</div>
        </div>
        <div className="bg-blue-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-blue-900">15</div>
          <div className="text-sm text-blue-700">Articles (24h)</div>
        </div>
        <div className="bg-purple-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-purple-900">0.73</div>
          <div className="text-sm text-purple-700">Sentiment Score</div>
        </div>
      </div>

      <div className="space-y-4">
        {[
          {
            title: `${entity.name} Reports Strong Q4 Earnings`,
            impact: 'positive',
            sentiment: 0.8,
            time: '2 hours ago',
            source: 'Reuters'
          },
          {
            title: `${entity.name} Announces Strategic Partnership`,
            impact: 'positive',
            sentiment: 0.6,
            time: '4 hours ago',
            source: 'Bloomberg'
          },
          {
            title: `Market Volatility Affects Tech Stocks`,
            impact: 'negative',
            sentiment: -0.3,
            time: '6 hours ago',
            source: 'Financial Times'
          }
        ].map((article, index) => (
          <NewsArticle key={index} article={article} />
        ))}
      </div>
    </div>
  );

  const NewsArticle = ({ article }) => (
    <div className="bg-white border rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <h4 className="font-medium text-gray-900 flex-1">{article.title}</h4>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ml-3 ${
          article.impact === 'positive' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
        }`}>
          {article.sentiment > 0 ? '+' : ''}{(article.sentiment * 10).toFixed(1)}
        </span>
      </div>
      <div className="flex justify-between items-center text-sm text-gray-500">
        <span>{article.source}</span>
        <span>{article.time}</span>
      </div>
    </div>
  );

  const ScenarioView = ({ entity }) => {
    const [scenarioType, setScenarioType] = useState('debt_reduction');
    const [scenarioValue, setScenarioValue] = useState(-10);
    const [scenarioResult, setScenarioResult] = useState(null);

    const runScenario = () => {
      // Simulate scenario analysis
      const baseScore = entity.score;
      let impact = 0;
      
      switch (scenarioType) {
        case 'debt_reduction':
          impact = Math.abs(scenarioValue) * 0.5;
          break;
        case 'margin_improvement':
          impact = scenarioValue * 2;
          break;
        case 'market_volatility':
          impact = -Math.abs(scenarioValue) * 0.3;
          break;
        default:
          impact = 0;
      }

      const newScore = baseScore + impact;
      setScenarioResult({
        originalScore: baseScore,
        newScore,
        impact,
        newRating: newScore > 800 ? 'AA' : newScore > 700 ? 'A' : newScore > 600 ? 'BBB' : 'BB'
      });
    };

    return (
      <div className="space-y-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-yellow-900 mb-3">Scenario Analysis</h3>
          <p className="text-yellow-800">
            Analyze how changes in key metrics would impact the credit score using our causal model.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Scenario Type
              </label>
              <select
                value={scenarioType}
                onChange={(e) => setScenarioType(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="debt_reduction">Debt Reduction</option>
                <option value="margin_improvement">Operating Margin Change</option>
                <option value="market_volatility">Market Volatility Change</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Change (%)
              </label>
              <input
                type="range"
                min={-50}
                max={50}
                value={scenarioValue}
                onChange={(e) => setScenarioValue(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-center text-sm text-gray-600 mt-1">
                {scenarioValue > 0 ? '+' : ''}{scenarioValue}%
              </div>
            </div>

            <button
              onClick={runScenario}
              className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              Run Scenario Analysis
            </button>
          </div>

          <div className="space-y-4">
            {scenarioResult && (
              <div className="bg-white border rounded-lg p-6">
                <h4 className="font-semibold text-gray-900 mb-4">Scenario Results</h4>
                
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Original Score:</span>
                    <span className="font-medium">{Math.round(scenarioResult.originalScore)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">New Score:</span>
                    <span className="font-medium">{Math.round(scenarioResult.newScore)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Impact:</span>
                    <span className={`font-medium ${
                      scenarioResult.impact > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {scenarioResult.impact > 0 ? '+' : ''}{Math.round(scenarioResult.impact)} points
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">New Rating:</span>
                    <span className="font-medium">{scenarioResult.newRating}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const MarketOverview = () => (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
      <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-xl p-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium opacity-90">Market Sentiment</h3>
          <Globe size={20} className="opacity-75" />
        </div>
        <div className="text-2xl font-bold capitalize">{marketData.market_sentiment}</div>
        <div className="text-sm opacity-75">Overall market conditions</div>
      </div>

      <div className="bg-gradient-to-br from-green-500 to-green-600 text-white rounded-xl p-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium opacity-90">Volatility Index</h3>
          <Activity size={20} className="opacity-75" />
        </div>
        <div className="text-2xl font-bold">{marketData.volatility_index}</div>
        <div className="text-sm opacity-75">Market stability measure</div>
      </div>

      <div className="bg-gradient-to-br from-purple-500 to-purple-600 text-white rounded-xl p-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium opacity-90">Tracked Entities</h3>
          <Users size={20} className="opacity-75" />
        </div>
        <div className="text-2xl font-bold">{marketData.total_entities}</div>
        <div className="text-sm opacity-75">Companies & sovereigns</div>
      </div>

      <div className="bg-gradient-to-br from-orange-500 to-orange-600 text-white rounded-xl p-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium opacity-90">Average Score</h3>
          <BarChart3 size={20} className="opacity-75" />
        </div>
        <div className="text-2xl font-bold">{marketData.average_score}</div>
        <div className="text-sm opacity-75">Market-wide average</div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Zap className="text-blue-600" size={32} />
                <h1 className="text-2xl font-bold text-gray-900">CredScope AI</h1>
              </div>
              <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                Real-time Credit Intelligence
              </span>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setIsLoading(!isLoading)}
                className={`p-2 rounded-lg transition-colors ${
                  isLoading ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                <RefreshCw size={20} className={isLoading ? 'animate-spin' : ''} />
              </button>
              
              <nav className="flex space-x-6">
                {['overview', 'market', 'alerts', 'settings'].map(tab => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-3 py-2 rounded-lg font-medium text-sm capitalize transition-colors ${
                      activeTab === tab
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </nav>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {selectedEntity ? (
          <EntityDetailView entity={selectedEntity} />
        ) : (
          <div className="space-y-8">
            <MarketOverview />
            
            {activeTab === 'overview' && (
              <div>
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold text-gray-900">Credit Score Overview</h2>
                  <div className="flex items-center space-x-2 text-sm text-gray-500">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span>Live updates active</span>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {entities.map(entity => (
                    <ScoreCard 
                      key={entity.entity_id} 
                      entity={entity} 
                      onClick={setSelectedEntity}
                    />
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'alerts' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-900">Active Alerts</h2>
                <div className="space-y-4">
                  {[
                    {
                      entity: 'XOM',
                      type: 'Score Drop',
                      severity: 'high',
                      message: 'Credit score dropped by 15 points in 24 hours',
                      time: '2 hours ago'
                    },
                    {
                      entity: 'TSLA',
                      type: 'News Impact',
                      severity: 'medium',
                      message: 'Positive earnings news boosting sentiment',
                      time: '4 hours ago'
                    }
                  ].map((alert, index) => (
                    <div key={index} className="bg-white border-l-4 border-red-400 p-4 rounded-r-lg shadow-sm">
                      <div className="flex justify-between items-start">
                        <div className="flex items-center space-x-3">
                          <AlertTriangle className="text-red-500" size={20} />
                          <div>
                            <h3 className="font-medium text-gray-900">{alert.entity} - {alert.type}</h3>
                            <p className="text-gray-600">{alert.message}</p>
                          </div>
                        </div>
                        <span className="text-sm text-gray-500">{alert.time}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default Dashboard;