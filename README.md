# Econometric Analysis of Housing Market Dynamics and Monetary Policy Transmission

This repository contains a comprehensive econometric investigation of the relationships between Federal Reserve monetary policy and United States housing market performance utilizing advanced time series methodologies applied to macroeconomic data spanning 2000-2024.

## Abstract

This research examines volatility dynamics, regime-switching behavior, jump risk characteristics, and policy transmission mechanisms within the US housing market through rigorous quantitative analysis. The study employs four complementary econometric frameworks to provide comprehensive insights into housing market behavior and monetary policy effectiveness.

### Research Objectives

- Quantification of asymmetric volatility responses in housing market returns
- Identification and characterization of distinct housing market regimes
- Assessment of discontinuous price movement risks and tail event probabilities  
- Analysis of temporal structures in Federal Reserve policy transmission mechanisms

## Methodology

### Econometric Frameworks

**1. Generalized Autoregressive Conditional Heteroskedasticity with Glosten-Jagannathan-Runkle Specification (GJR-GARCH(1,1))**
- Asymmetric volatility modeling with leverage effects
- Conditional heteroskedasticity estimation
- Federal funds rate incorporation as external regressor

**2. Bayesian Markov Regime Switching Model**
- Markov Chain Monte Carlo estimation procedures
- Regime-dependent parameter estimation
- Transition probability matrix calculation
- Policy variable integration as regime-dependent covariates

**3. Merton Jump-Diffusion Model**
- Discontinuous price movement characterization
- Jump intensity and magnitude distribution estimation
- Comprehensive risk metric computation including Value-at-Risk calculations

**4. Transfer Function Model with Distributed Lag Structure**
- Twelve-lag specification for policy transmission analysis
- Impulse response function estimation
- Dynamic relationship quantification between monetary policy and housing returns

## Data

### Primary Data Sources

**Case-Shiller US National Home Price Index (CSUSHPINSA)**
- Temporal Coverage: January 2000 - December 2024
- Frequency: Monthly
- Observations: 302
- Source: S&P Dow Jones Indices

**Zillow Home Value Index Regional Dataset**
- Temporal Coverage: January 2000 - March 2025
- Frequency: Monthly  
- Geographic Coverage: 556 regions
- Temporal Dimensions: 308 periods

**Federal Funds Effective Rate**
- Temporal Coverage: January 2000 - December 2024
- Frequency: Monthly
- Observations: 300
- Source: Federal Reserve Economic Data (FRED)

## System Architecture

### Full-Stack Platform Overview

This project implements a comprehensive housing market analysis platform featuring:
- **RESTful API Backend**: Scalable microservices architecture serving econometric model endpoints
- **PostgreSQL Database**: Optimized relational database managing 270,000+ housing market records
- **Containerized Deployment**: Docker-based containerization for consistent deployment environments
- **CI/CD Pipeline**: Automated testing, building, and deployment workflows
- **JWT Authentication**: Secure token-based authentication system for API access control

### Technology Stack

**Backend Services**
```
Python >=3.8
FastAPI >=0.68.0
PostgreSQL >=13.0
SQLAlchemy >=1.4.0
Alembic >=1.7.0
Redis >=6.2.0
Celery >=5.2.0
```

**Econometric Computing**
```
pandas >=1.5.0
numpy >=1.21.0
arch >=5.3.0
pymc >=5.0.0
statsmodels >=0.13.0
scipy >=1.9.0
scikit-learn >=1.1.0
```

**Infrastructure & DevOps**
```
Docker >=20.10.0
Docker Compose >=2.0.0
PostgreSQL >=13.0
Redis >=6.2.0
Nginx >=1.20.0
```

**Authentication & Security**
```
PyJWT >=2.4.0
passlib >=1.7.4
python-multipart >=0.0.5
bcrypt >=3.2.0
```

### Installation and Deployment

#### Local Development Setup

```bash
# Clone repository
git clone https://github.com/username/econometric-housing-analysis.git
cd econometric-housing-analysis

# Environment setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Database initialization
docker-compose up -d postgres redis
alembic upgrade head

# Data migration
python scripts/load_housing_data.py --source data/
```

#### Production Deployment with Docker

```bash
# Build and deploy containerized application
docker-compose -f docker-compose.prod.yml up -d

# Database migration in production
docker-compose exec api alembic upgrade head

# Load production datasets
docker-compose exec api python scripts/migrate_data.py
```

### API Endpoints

#### Authentication
```
POST /auth/login          # JWT token generation
POST /auth/refresh        # Token refresh
POST /auth/logout         # Token invalidation
```

#### Econometric Analysis
```
GET  /api/v1/models/garch          # GJR-GARCH model results
GET  /api/v1/models/regime         # Regime switching analysis
GET  /api/v1/models/jump           # Jump-diffusion parameters
GET  /api/v1/models/transfer       # Transfer function results
POST /api/v1/forecast              # Generate forecasts
```

#### Data Management
```
GET  /api/v1/housing/records       # Housing market data retrieval
GET  /api/v1/fed/rates             # Federal funds rate data
POST /api/v1/data/upload           # Bulk data ingestion
GET  /api/v1/data/status           # Processing status monitoring
```

### Database Schema

#### Core Tables
```sql
-- Housing market records (270K+ records)
housing_data (
    id SERIAL PRIMARY KEY,
    region_id INTEGER,
    observation_date DATE,
    price_index DECIMAL(12,6),
    return_rate DECIMAL(8,6),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Federal funds rate data
fed_rates (
    id SERIAL PRIMARY KEY,
    observation_date DATE,
    rate DECIMAL(6,4),
    rate_change DECIMAL(6,4),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model results storage
model_results (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50),
    parameters JSONB,
    diagnostics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Containerization Architecture

#### Docker Services
```yaml
# docker-compose.yml structure
services:
  api:          # FastAPI application server
  postgres:     # PostgreSQL database
  redis:        # Redis cache and message broker
  celery:       # Asynchronous task processing
  nginx:        # Reverse proxy and load balancer
```

#### Security Implementation
- **JWT Authentication**: Stateless token-based authentication with configurable expiration
- **Password Hashing**: bcrypt-based secure password storage
- **CORS Configuration**: Cross-origin resource sharing controls
- **Rate Limiting**: Request throttling and abuse prevention
- **Input Validation**: Comprehensive request validation using Pydantic models

## Analytical Framework

### Data Processing Pipeline

- Multi-source data integration and temporal alignment
- Feature engineering with lagged variable construction
- Missing value imputation and data validation protocols
- Statistical transformation and normalization procedures

### Model Estimation Procedures

- **Parameter Estimation**: Maximum likelihood estimation and Bayesian inference methods
- **Model Validation**: Comprehensive diagnostic testing protocols
- **Forecasting Framework**: Out-of-sample prediction and uncertainty quantification
- **Visualization**: Statistical graphics and model output presentation

### Risk Analytics Module

- Volatility persistence coefficient estimation
- Regime transition probability matrices
- Jump event frequency and magnitude distributions
- Policy transmission lag structure identification

## Results Framework

### Model Outputs

**Volatility Analysis**
- Conditional volatility estimates with confidence intervals
- Asymmetry parameter quantification
- Persistence coefficient determination

**Regime Classification**
- Market state identification with associated probabilities
- Regime-dependent statistical moments
- Transition timing and duration analysis

**Risk Assessment**
- Value-at-Risk estimates across multiple confidence levels
- Expected shortfall calculations
- Tail event probability distributions

**Policy Transmission Analysis**
- Impulse response function estimates
- Peak response timing identification
- Long-term multiplier effects quantification

## Model Validation

### Diagnostic Procedures

- **Serial Correlation Testing**: Ljung-Box test statistics
- **Normality Assessment**: Jarque-Bera test procedures
- **Heteroskedasticity Evaluation**: Autoregressive Conditional Heteroskedasticity Lagrange Multiplier tests
- **Bayesian Model Diagnostics**: Gelman-Rubin convergence statistics and effective sample size calculations
- **Forecasting Accuracy**: Out-of-sample prediction evaluation metrics

## Repository Structure

```
econometric-housing-analysis/
├── api/
│   ├── routers/
│   │   ├── auth.py
│   │   ├── models.py
│   │   ├── housing.py
│   │   └── forecasting.py
│   ├── models/
│   │   ├── database.py
│   │   ├── schemas.py
│   │   └── crud.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── dependencies.py
│   └── main.py
├── econometric/
│   ├── models/
│   │   ├── gjr_garch.py
│   │   ├── regime_switching.py
│   │   ├── jump_diffusion.py
│   │   └── transfer_function.py
│   ├── data_processing/
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   └── feature_engineering.py
│   └── analytics/
│       ├── risk_metrics.py
│       ├── forecasting.py
│       └── diagnostics.py
├── data/
│   ├── raw/
│   │   ├── CSUSHPINSA.csv
│   │   ├── housing_data_filtered_regions.csv
│   │   └── fed_rate_clean_2000_20241.csv
│   └── processed/
├── database/
│   ├── migrations/
│   └── seeds/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── load/
├── scripts/
│   ├── load_housing_data.py
│   ├── migrate_data.py
│   └── backup_database.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── docs/
│   ├── api_documentation.md
│   ├── deployment_guide.md
│   └── user_manual.md
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── LICENSE
```

## CI/CD Pipeline

### Automated Workflows

**Continuous Integration** (`.github/workflows/ci.yml`)
```yaml
# Automated testing on push/PR
- Unit test execution with pytest
- Integration test validation
- Code quality assessment with flake8
- Security vulnerability scanning
- Docker image building and testing
```

**Continuous Deployment** (`.github/workflows/deploy.yml`)
```yaml
# Production deployment automation
- Environment-specific configuration management
- Database migration execution
- Rolling deployment with zero downtime
- Health check validation
- Rollback procedures
```

### Testing Framework

**Test Coverage**
- **Unit Tests**: Individual component validation (>90% coverage)
- **Integration Tests**: API endpoint and database interaction testing
- **Load Tests**: Performance validation under simulated traffic
- **Model Validation Tests**: Econometric model accuracy and convergence testing

### Performance Optimization

**Database Optimization**
- Indexed queries for time-series data retrieval
- Connection pooling for concurrent request handling
- Read replica configuration for analytical workloads
- Partitioning strategy for large datasets

**Caching Strategy**
- Redis-based caching for frequently accessed model results
- Query result caching with configurable TTL
- Session management and token storage

**Asynchronous Processing**
- Celery task queue for computationally intensive model estimation
- Background job processing for large dataset analysis
- Real-time progress monitoring and status updates

## Research Applications

### Academic Research Domain
- Housing finance and real estate economics research
- Monetary policy transmission mechanism studies
- Advanced econometric methodology development


## Contribution Guidelines

Contributions to this research project are welcomed through:
- Model specification enhancements
- Alternative econometric methodology implementations
- Computational efficiency improvements
- Documentation and methodology clarification

Please submit pull requests or create issues for proposed modifications.

## Academic References

### Theoretical Foundations
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779-1801.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1-2), 125-144.

### Data Sources
- S&P Dow Jones Indices: Case-Shiller Home Price Index Methodology
- Zillow Research: Home Value Index Technical Documentation
- Federal Reserve Bank of St. Louis: Federal Reserve Economic Data (FRED)

## License

This project is distributed under the MIT License. See LICENSE file for complete terms and conditions.

## Contact Information

For inquiries regarding methodology, implementation details, or research collaboration opportunities, please contact the project maintainer through the repository's issue tracking system.

---

*This research was conducted as part of advanced econometric analysis in housing finance and monetary economics. All methodologies follow established academic standards and best practices in financial econometrics.*
