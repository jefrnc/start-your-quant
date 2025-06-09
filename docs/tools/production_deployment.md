# Production Deployment para Trading Algorítmico

## Introducción: De Backtesting a Producción

El deployment exitoso de estrategias algorítmicas requiere una infraestructura robusta que maneje alta disponibilidad, baja latencia, monitoreo continuo y recuperación automática. Este documento cubre la arquitectura completa para deployment profesional.

## Architecture de Production

### Core Infrastructure

```python
import asyncio
import logging
import os
import sys
import time
import json
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import docker
import kubernetes
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import redis
import psycopg2
from sqlalchemy import create_engine
import boto3

@dataclass
class DeploymentConfig:
    """Configuración de deployment"""
    environment: str  # dev, staging, prod
    strategy_name: str
    version: str
    replicas: int
    resources: Dict[str, str]
    health_check_interval: int
    max_memory_mb: int
    max_cpu_cores: float
    
    # Database config
    database_url: str
    redis_url: str
    
    # Monitoring config
    metrics_port: int
    log_level: str
    
    # Trading config
    broker_config: Dict[str, Any]
    risk_limits: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ProductionInfrastructure:
    """Infraestructura de producción"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.metrics = self._setup_metrics()
        self.database = self._setup_database()
        self.cache = self._setup_cache()
        self.health_status = "starting"
        
        # Service discovery
        self.services = {}
        
        # Circuit breakers
        self.circuit_breakers = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Configura logging estructurado"""
        logger = logging.getLogger(f"trading-{self.config.strategy_name}")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"service": "%(name)s", "message": "%(message)s", '
            '"strategy": "' + self.config.strategy_name + '", '
            '"environment": "' + self.config.environment + '"}'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler for persistent logs
        if self.config.environment == 'prod':
            file_handler = logging.FileHandler(
                f'/var/log/trading/{self.config.strategy_name}.log'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_metrics(self) -> Dict[str, Any]:
        """Configura métricas de Prometheus"""
        metrics = {
            'trades_total': Counter(
                'trading_trades_total',
                'Total number of trades',
                ['strategy', 'symbol', 'side', 'status']
            ),
            'pnl_total': Gauge(
                'trading_pnl_total',
                'Total PnL',
                ['strategy', 'symbol']
            ),
            'latency_seconds': Histogram(
                'trading_latency_seconds',
                'Trading operation latency',
                ['operation']
            ),
            'positions_count': Gauge(
                'trading_positions_count',
                'Number of open positions',
                ['strategy']
            ),
            'risk_utilization': Gauge(
                'trading_risk_utilization',
                'Risk utilization percentage',
                ['risk_type', 'strategy']
            ),
            'health_status': Gauge(
                'trading_health_status',
                'Service health status (1=healthy, 0=unhealthy)',
                ['service', 'strategy']
            )
        }
        
        # Start metrics server
        prometheus_client.start_http_server(self.config.metrics_port)
        
        return metrics
    
    def _setup_database(self):
        """Configura conexión a base de datos"""
        try:
            engine = create_engine(
                self.config.database_url,
                pool_size=10,
                pool_recycle=3600,
                pool_pre_ping=True
            )
            
            # Test connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            
            self.logger.info("Database connection established")
            return engine
            
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def _setup_cache(self):
        """Configura Redis para caching"""
        try:
            cache = redis.from_url(
                self.config.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            cache.ping()
            
            self.logger.info("Redis cache connection established")
            return cache
            
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise
    
    async def deploy(self) -> bool:
        """Deploya la estrategia en producción"""
        try:
            self.logger.info(f"Starting deployment of {self.config.strategy_name} v{self.config.version}")
            
            # 1. Pre-deployment checks
            if not await self._pre_deployment_checks():
                return False
            
            # 2. Deploy containers
            if not await self._deploy_containers():
                return False
            
            # 3. Setup monitoring
            await self._setup_monitoring()
            
            # 4. Start health checks
            await self._start_health_checks()
            
            # 5. Gradual traffic routing
            await self._gradual_traffic_routing()
            
            # 6. Post-deployment validation
            if not await self._post_deployment_validation():
                await self._rollback()
                return False
            
            self.health_status = "healthy"
            self.logger.info("Deployment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            await self._rollback()
            return False
    
    async def _pre_deployment_checks(self) -> bool:
        """Verificaciones pre-deployment"""
        checks = {
            'database_connectivity': self._check_database_connectivity(),
            'cache_connectivity': self._check_cache_connectivity(),
            'broker_connectivity': await self._check_broker_connectivity(),
            'market_data_connectivity': await self._check_market_data_connectivity(),
            'resource_availability': self._check_resource_availability(),
            'configuration_validation': self._validate_configuration()
        }
        
        failed_checks = []
        for check_name, check_result in checks.items():
            if not check_result:
                failed_checks.append(check_name)
        
        if failed_checks:
            self.logger.error(f"Pre-deployment checks failed: {failed_checks}")
            return False
        
        self.logger.info("All pre-deployment checks passed")
        return True
    
    async def _deploy_containers(self) -> bool:
        """Deploya containers usando Kubernetes"""
        try:
            # Load Kubernetes config
            kubernetes.config.load_incluster_config()
            v1 = kubernetes.client.AppsV1Api()
            
            # Create deployment manifest
            deployment_manifest = self._create_deployment_manifest()
            
            # Deploy
            namespace = f"trading-{self.config.environment}"
            v1.create_namespaced_deployment(
                namespace=namespace,
                body=deployment_manifest
            )
            
            # Wait for deployment to be ready
            await self._wait_for_deployment_ready(namespace, deployment_manifest['metadata']['name'])
            
            self.logger.info("Container deployment successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Container deployment failed: {e}")
            return False
    
    def _create_deployment_manifest(self) -> Dict[str, Any]:
        """Crea manifest de Kubernetes"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.config.strategy_name}-{self.config.version}",
                "labels": {
                    "app": self.config.strategy_name,
                    "version": self.config.version,
                    "environment": self.config.environment
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.config.strategy_name,
                        "version": self.config.version
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.strategy_name,
                            "version": self.config.version
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": self.config.strategy_name,
                            "image": f"trading/{self.config.strategy_name}:{self.config.version}",
                            "ports": [
                                {"containerPort": 8080, "name": "http"},
                                {"containerPort": self.config.metrics_port, "name": "metrics"}
                            ],
                            "resources": {
                                "limits": {
                                    "memory": f"{self.config.max_memory_mb}Mi",
                                    "cpu": str(self.config.max_cpu_cores)
                                },
                                "requests": {
                                    "memory": f"{self.config.max_memory_mb//2}Mi",
                                    "cpu": str(self.config.max_cpu_cores/2)
                                }
                            },
                            "env": [
                                {"name": "ENVIRONMENT", "value": self.config.environment},
                                {"name": "STRATEGY_NAME", "value": self.config.strategy_name},
                                {"name": "VERSION", "value": self.config.version}
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }

class TradingStrategyService:
    """Servicio de estrategia de trading en producción"""
    
    def __init__(self, infrastructure: ProductionInfrastructure):
        self.infrastructure = infrastructure
        self.logger = infrastructure.logger
        self.metrics = infrastructure.metrics
        
        # Strategy components
        self.data_manager = None
        self.risk_manager = None
        self.order_manager = None
        self.strategy_engine = None
        
        # Service state
        self.is_running = False
        self.last_heartbeat = None
        self.error_count = 0
        
    async def start(self):
        """Inicia el servicio de trading"""
        try:
            self.logger.info("Starting trading strategy service")
            
            # 1. Initialize components
            await self._initialize_components()
            
            # 2. Start data feeds
            await self._start_data_feeds()
            
            # 3. Start strategy engine
            await self._start_strategy_engine()
            
            # 4. Start monitoring
            await self._start_internal_monitoring()
            
            self.is_running = True
            self.logger.info("Trading strategy service started successfully")
            
            # 5. Main service loop
            await self._service_loop()
            
        except Exception as e:
            self.logger.error(f"Service startup failed: {e}")
            await self._emergency_shutdown()
            raise
    
    async def _initialize_components(self):
        """Inicializa componentes del servicio"""
        
        # Data manager
        self.data_manager = ProductionDataManager(
            config=self.infrastructure.config.broker_config,
            cache=self.infrastructure.cache
        )
        
        # Risk manager
        self.risk_manager = ProductionRiskManager(
            limits=self.infrastructure.config.risk_limits,
            database=self.infrastructure.database
        )
        
        # Order manager
        self.order_manager = ProductionOrderManager(
            broker_config=self.infrastructure.config.broker_config,
            database=self.infrastructure.database
        )
        
        # Strategy engine
        self.strategy_engine = ProductionStrategyEngine(
            strategy_name=self.infrastructure.config.strategy_name,
            data_manager=self.data_manager,
            risk_manager=self.risk_manager,
            order_manager=self.order_manager
        )
        
        self.logger.info("All components initialized")
    
    async def _service_loop(self):
        """Loop principal del servicio"""
        while self.is_running:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Check system health
                health_status = await self._check_system_health()
                if not health_status['healthy']:
                    await self._handle_health_issues(health_status)
                
                # Update metrics
                self._update_service_metrics()
                
                # Process any pending orders
                await self._process_pending_orders()
                
                # Check for emergency conditions
                emergency_status = await self._check_emergency_conditions()
                if emergency_status['emergency']:
                    await self._handle_emergency(emergency_status)
                
                # Sleep before next iteration
                await asyncio.sleep(1)  # 1 second intervals
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Service loop error: {e}")
                
                if self.error_count > 10:  # Too many errors
                    await self._emergency_shutdown()
                    break
                
                await asyncio.sleep(5)  # Wait before retry
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Verifica salud del sistema"""
        health_checks = {
            'data_feed_healthy': await self.data_manager.health_check(),
            'risk_manager_healthy': self.risk_manager.health_check(),
            'order_manager_healthy': await self.order_manager.health_check(),
            'database_healthy': self._check_database_health(),
            'cache_healthy': self._check_cache_health(),
            'memory_usage_ok': self._check_memory_usage(),
            'cpu_usage_ok': self._check_cpu_usage()
        }
        
        failed_checks = [check for check, status in health_checks.items() if not status]
        
        return {
            'healthy': len(failed_checks) == 0,
            'failed_checks': failed_checks,
            'health_score': (len(health_checks) - len(failed_checks)) / len(health_checks)
        }
    
    async def _handle_emergency(self, emergency_status: Dict[str, Any]):
        """Maneja condiciones de emergencia"""
        emergency_type = emergency_status['type']
        
        self.logger.critical(f"Emergency condition detected: {emergency_type}")
        
        if emergency_type == 'risk_breach':
            # Close all positions immediately
            await self.order_manager.close_all_positions()
            
        elif emergency_type == 'data_feed_failure':
            # Switch to backup data feed
            await self.data_manager.switch_to_backup_feed()
            
        elif emergency_type == 'broker_connectivity':
            # Attempt to reconnect and halt new orders
            await self.order_manager.halt_new_orders()
            await self.order_manager.reconnect()
            
        elif emergency_type == 'system_overload':
            # Reduce processing load
            await self.strategy_engine.reduce_processing_load()
        
        # Update emergency metrics
        self.metrics['health_status'].labels(
            service='emergency_handler',
            strategy=self.infrastructure.config.strategy_name
        ).set(0)

class ProductionMonitoringSystem:
    """Sistema de monitoreo para producción"""
    
    def __init__(self, infrastructure: ProductionInfrastructure):
        self.infrastructure = infrastructure
        self.logger = infrastructure.logger
        self.metrics = infrastructure.metrics
        
        # Alerting thresholds
        self.alert_thresholds = {
            'max_latency_ms': 100,
            'max_error_rate': 0.05,  # 5%
            'min_uptime': 0.99,      # 99%
            'max_memory_usage': 0.85, # 85%
            'max_cpu_usage': 0.80     # 80%
        }
        
        # Alert channels
        self.alert_channels = []
        
    async def start_monitoring(self):
        """Inicia sistema de monitoreo"""
        monitoring_tasks = [
            self._monitor_performance_metrics(),
            self._monitor_business_metrics(),
            self._monitor_system_resources(),
            self._monitor_error_rates(),
            self._monitor_sla_compliance()
        ]
        
        await asyncio.gather(*monitoring_tasks)
    
    async def _monitor_performance_metrics(self):
        """Monitorea métricas de performance"""
        while True:
            try:
                # Latency monitoring
                latency_metrics = await self._collect_latency_metrics()
                
                for operation, latency in latency_metrics.items():
                    self.metrics['latency_seconds'].labels(operation=operation).observe(latency)
                    
                    if latency * 1000 > self.alert_thresholds['max_latency_ms']:
                        await self._send_alert({
                            'type': 'high_latency',
                            'operation': operation,
                            'value': latency * 1000,
                            'threshold': self.alert_thresholds['max_latency_ms']
                        })
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Envía alerta a canales configurados"""
        alert_message = {
            'timestamp': datetime.now().isoformat(),
            'strategy': self.infrastructure.config.strategy_name,
            'environment': self.infrastructure.config.environment,
            'alert': alert
        }
        
        # Log alert
        self.logger.warning(f"ALERT: {alert}")
        
        # Send to configured channels
        for channel in self.alert_channels:
            try:
                await channel.send_alert(alert_message)
            except Exception as e:
                self.logger.error(f"Failed to send alert to {channel}: {e}")

class BlueGreenDeployment:
    """Implementación de Blue-Green deployment"""
    
    def __init__(self, infrastructure: ProductionInfrastructure):
        self.infrastructure = infrastructure
        self.logger = infrastructure.logger
        
    async def deploy_new_version(self, new_version: str) -> bool:
        """Deploya nueva versión usando Blue-Green strategy"""
        try:
            self.logger.info(f"Starting Blue-Green deployment to version {new_version}")
            
            # 1. Deploy green environment
            green_config = self._create_green_config(new_version)
            green_infrastructure = ProductionInfrastructure(green_config)
            
            if not await green_infrastructure.deploy():
                self.logger.error("Green environment deployment failed")
                return False
            
            # 2. Warm up green environment
            await self._warm_up_environment(green_infrastructure)
            
            # 3. Run smoke tests
            if not await self._run_smoke_tests(green_infrastructure):
                self.logger.error("Smoke tests failed")
                await self._cleanup_green_environment(green_infrastructure)
                return False
            
            # 4. Switch traffic to green
            await self._switch_traffic_to_green(green_infrastructure)
            
            # 5. Monitor green environment
            if not await self._monitor_green_environment(green_infrastructure):
                self.logger.error("Green environment monitoring failed")
                await self._rollback_to_blue()
                return False
            
            # 6. Cleanup blue environment
            await self._cleanup_blue_environment()
            
            self.logger.info("Blue-Green deployment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Blue-Green deployment failed: {e}")
            await self._rollback_to_blue()
            return False
    
    def _create_green_config(self, new_version: str) -> DeploymentConfig:
        """Crea configuración para ambiente Green"""
        green_config = DeploymentConfig(
            environment=f"{self.infrastructure.config.environment}-green",
            strategy_name=self.infrastructure.config.strategy_name,
            version=new_version,
            replicas=self.infrastructure.config.replicas,
            resources=self.infrastructure.config.resources,
            health_check_interval=self.infrastructure.config.health_check_interval,
            max_memory_mb=self.infrastructure.config.max_memory_mb,
            max_cpu_cores=self.infrastructure.config.max_cpu_cores,
            database_url=self.infrastructure.config.database_url,
            redis_url=self.infrastructure.config.redis_url,
            metrics_port=self.infrastructure.config.metrics_port + 1000,  # Different port
            log_level=self.infrastructure.config.log_level,
            broker_config=self.infrastructure.config.broker_config.copy(),
            risk_limits=self.infrastructure.config.risk_limits.copy()
        )
        
        return green_config

class AutoScalingManager:
    """Manager de auto-scaling para strategies"""
    
    def __init__(self, infrastructure: ProductionInfrastructure):
        self.infrastructure = infrastructure
        self.logger = infrastructure.logger
        
        # Scaling parameters
        self.min_replicas = 1
        self.max_replicas = 10
        self.target_cpu_utilization = 0.70
        self.target_memory_utilization = 0.75
        self.scale_up_threshold = 0.85
        self.scale_down_threshold = 0.50
        
    async def start_autoscaling(self):
        """Inicia auto-scaling"""
        while True:
            try:
                # Get current metrics
                current_metrics = await self._get_current_metrics()
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(current_metrics)
                
                if scaling_decision['action'] != 'none':
                    await self._execute_scaling(scaling_decision)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(60)
    
    def _make_scaling_decision(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Toma decisión de scaling basada en métricas"""
        current_replicas = metrics['current_replicas']
        cpu_utilization = metrics['cpu_utilization']
        memory_utilization = metrics['memory_utilization']
        request_rate = metrics['request_rate']
        
        # Scale up conditions
        if (cpu_utilization > self.scale_up_threshold or 
            memory_utilization > self.scale_up_threshold):
            
            if current_replicas < self.max_replicas:
                new_replicas = min(current_replicas + 1, self.max_replicas)
                return {
                    'action': 'scale_up',
                    'current_replicas': current_replicas,
                    'new_replicas': new_replicas,
                    'reason': f'High resource utilization: CPU={cpu_utilization:.2f}, Memory={memory_utilization:.2f}'
                }
        
        # Scale down conditions
        elif (cpu_utilization < self.scale_down_threshold and 
              memory_utilization < self.scale_down_threshold):
            
            if current_replicas > self.min_replicas:
                new_replicas = max(current_replicas - 1, self.min_replicas)
                return {
                    'action': 'scale_down',
                    'current_replicas': current_replicas,
                    'new_replicas': new_replicas,
                    'reason': f'Low resource utilization: CPU={cpu_utilization:.2f}, Memory={memory_utilization:.2f}'
                }
        
        return {'action': 'none'}

# Usage Example
async def main():
    """Ejemplo completo de deployment en producción"""
    
    # 1. Create deployment configuration
    config = DeploymentConfig(
        environment="prod",
        strategy_name="gap_and_go",
        version="1.2.3",
        replicas=3,
        resources={"cpu": "1000m", "memory": "2Gi"},
        health_check_interval=30,
        max_memory_mb=2048,
        max_cpu_cores=1.0,
        database_url="postgresql://user:pass@db:5432/trading",
        redis_url="redis://redis:6379/0",
        metrics_port=9090,
        log_level="INFO",
        broker_config={
            "api_key": "your_key",
            "api_secret": "your_secret",
            "base_url": "https://paper-api.alpaca.markets"
        },
        risk_limits={
            "max_position_size": 0.05,
            "max_daily_loss": 0.02,
            "max_portfolio_var": 0.03
        }
    )
    
    # 2. Initialize infrastructure
    infrastructure = ProductionInfrastructure(config)
    
    # 3. Deploy to production
    deployment_success = await infrastructure.deploy()
    
    if deployment_success:
        print("✅ Deployment successful!")
        
        # 4. Start trading service
        trading_service = TradingStrategyService(infrastructure)
        
        # 5. Start monitoring
        monitoring_system = ProductionMonitoringSystem(infrastructure)
        
        # 6. Start auto-scaling
        autoscaling_manager = AutoScalingManager(infrastructure)
        
        # Run all services concurrently
        await asyncio.gather(
            trading_service.start(),
            monitoring_system.start_monitoring(),
            autoscaling_manager.start_autoscaling()
        )
    else:
        print("❌ Deployment failed!")

if __name__ == "__main__":
    asyncio.run(main())
```

## Docker Configuration

### Multi-Stage Dockerfile

```dockerfile
# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r trading && useradd -r -g trading trading

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/trading/.local

# Set PATH
ENV PATH=/home/trading/.local/bin:$PATH

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy application code
COPY --chown=trading:trading . .

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Start application
CMD ["python", "-m", "trading.main"]
```

### Docker Compose for Development

```yaml
version: '3.8'

services:
  trading-strategy:
    build:
      context: .
      target: production
    environment:
      - ENVIRONMENT=dev
      - DATABASE_URL=postgresql://trading:password@postgres:5432/trading_dev
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8080:8080"
      - "9090:9090"
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/var/log/trading
    restart: unless-stopped
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: trading_dev
      POSTGRES_USER: trading
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
      
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards
      
volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

## Kubernetes Deployment

### Complete K8s Manifests

```yaml
# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: trading-prod
  labels:
    environment: production
    
---
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-config
  namespace: trading-prod
data:
  environment: "prod"
  log_level: "INFO"
  metrics_port: "9090"
  
---
# Secret
apiVersion: v1
kind: Secret
metadata:
  name: trading-secrets
  namespace: trading-prod
type: Opaque
stringData:
  database-url: "postgresql://user:pass@postgres:5432/trading"
  redis-url: "redis://redis:6379/0"
  broker-api-key: "your-api-key"
  broker-api-secret: "your-api-secret"
  
---
# Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-strategy
  namespace: trading-prod
  labels:
    app: trading-strategy
    version: "1.0.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-strategy
  template:
    metadata:
      labels:
        app: trading-strategy
        version: "1.0.0"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: trading-strategy
        image: trading/strategy:1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: trading-config
              key: environment
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: redis-url
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
          
---
# Service
apiVersion: v1
kind: Service
metadata:
  name: trading-strategy-service
  namespace: trading-prod
  labels:
    app: trading-strategy
spec:
  selector:
    app: trading-strategy
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
    
---
# HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: trading-strategy-hpa
  namespace: trading-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trading-strategy
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
name: Trading Strategy CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: trading_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
        
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: pypa/gh-action-pip-audit@v1.0.8
      
    - name: Run Bandit security check
      run: |
        pip install bandit
        bandit -r src/
        
  build-and-push:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        
  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: staging
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        # Update Kubernetes manifests with new image
        sed -i "s|trading/strategy:.*|trading/strategy:${{ github.sha }}|" k8s/staging/deployment.yaml
        
        # Apply to staging cluster
        kubectl apply -f k8s/staging/ --validate=false
        
        # Wait for rollout
        kubectl rollout status deployment/trading-strategy -n trading-staging
        
  deploy-production:
    needs: [build-and-push, deploy-staging]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        # Blue-Green deployment
        ./scripts/blue-green-deploy.sh ${{ github.sha }}
```

---

*El deployment exitoso en producción requiere una combinación de infraestructura robusta, monitoreo comprehensivo, y procesos automatizados. Esta arquitectura asegura alta disponibilidad, escalabilidad automática, y recovery rápido ante fallas.*