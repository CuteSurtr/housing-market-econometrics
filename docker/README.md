# Housing Market Econometrics - Docker Setup

This directory contains the complete Docker configuration for the Housing Market Econometrics project, including development and production environments.

## Quick Start

### Development Environment

1. **Navigate to the docker directory:**
   ```bash
   cd docker
   ```

2. **Copy and configure environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Start the development environment:**
   ```bash
   docker-compose up -d
   ```

4. **Run database migrations:**
   ```bash
   docker-compose exec api alembic upgrade head
   ```

5. **Load sample data:**
   ```bash
   docker-compose exec api python scripts/load_housing_data.py --source data/
   ```

### Production Environment

1. **Configure production environment:**
   ```bash
   cp env.example .env
   # Edit .env with production values
   ```

2. **Start production environment:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Run database migrations:**
   ```bash
   docker-compose -f docker-compose.prod.yml exec api alembic upgrade head
   ```

## Services Overview

### Development Services (`docker-compose.yml`)

- **PostgreSQL** (port 5432): Main database
- **Redis** (port 6379): Cache and message broker
- **FastAPI** (port 8000): Main application
- **Celery Worker**: Background task processing
- **Celery Beat**: Scheduled task scheduler
- **Flower** (port 5555): Celery monitoring dashboard

### Production Services (`docker-compose.prod.yml`)

- **Nginx** (ports 80, 443): Reverse proxy and load balancer
- **PostgreSQL**: Production database
- **Redis**: Production cache with authentication
- **FastAPI**: Production application
- **Celery Worker**: Production background processing
- **Celery Beat**: Production scheduled tasks
- **Prometheus** (port 9090): Metrics collection
- **Grafana** (port 3000): Monitoring dashboards

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Database
POSTGRES_DB=housing_econometrics
POSTGRES_USER=econometrics_user
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_PASSWORD=your_redis_password

# Application
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Monitoring
GRAFANA_PASSWORD=admin
```

### Network Configuration

All services communicate through the `econometrics_network` bridge network.

## Monitoring

### Development Monitoring

- **Flower Dashboard**: http://localhost:5555
  - Monitor Celery tasks
  - View task history and performance

### Production Monitoring

- **Prometheus**: http://localhost:9090
  - View metrics and alerts
  - Query performance data

- **Grafana**: http://localhost:3000
  - Default credentials: admin/admin
  - View dashboards and visualizations

## Useful Commands

### Development

```bash
# View logs
docker-compose logs -f api

# Access database
docker-compose exec postgres psql -U econometrics_user -d housing_econometrics

# Access Redis
docker-compose exec redis redis-cli

# Run tests
docker-compose exec api pytest

# Restart services
docker-compose restart api
```

### Production

```bash
# View logs
docker-compose -f docker-compose.prod.yml logs -f api

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale celery_worker=3

# Backup database
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U econometrics_user housing_econometrics > backup.sql

# Monitor resources
docker stats
```

### General

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Rebuild images
docker-compose build --no-cache

# Update dependencies
docker-compose exec api pip install -r requirements.txt
```

## Health Checks

All services include health checks:

- **PostgreSQL**: `pg_isready` command
- **Redis**: `redis-cli ping` command
- **FastAPI**: HTTP health endpoint
- **Celery**: Internal health monitoring

## Security

### Development
- Default passwords for quick setup
- No SSL/TLS encryption
- Debug mode enabled

### Production
- Strong passwords required
- SSL/TLS encryption with Nginx
- Security headers enabled
- Rate limiting configured
- Non-root user in containers

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose files
2. **Permission errors**: Ensure proper file permissions
3. **Memory issues**: Increase Docker memory allocation
4. **Database connection**: Check PostgreSQL health status

### Debug Commands

```bash
# Check service status
docker-compose ps

# View detailed logs
docker-compose logs --tail=100 api

# Access container shell
docker-compose exec api bash

# Check network connectivity
docker-compose exec api ping postgres
```

## Performance Tuning

### Development
- Single worker processes
- Minimal resource allocation
- Debug logging enabled

### Production
- Multiple Celery workers
- Optimized resource allocation
- Production logging
- Monitoring and alerting

## Backup and Recovery

### Database Backup
```bash
# Create backup
docker-compose exec postgres pg_dump -U econometrics_user housing_econometrics > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
docker-compose exec -T postgres psql -U econometrics_user housing_econometrics < backup.sql
```

### Volume Backup
```bash
# Backup volumes
docker run --rm -v housing-market-econometrics_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v housing-market-econometrics_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_backup.tar.gz -C /data
```

## Next Steps

1. Configure SSL certificates for production
2. Set up automated backups
3. Configure monitoring alerts
4. Implement CI/CD pipeline
5. Set up load balancing for high availability

For more information, see the main project README and documentation. 