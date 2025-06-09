# Análisis de Redes en Finanzas

## Introducción: De Pixeles a Empresas

Los enfoques de redes en finanzas toman prestadas técnicas del análisis de imágenes y las adaptan para entender relaciones entre activos financieros. Similar a como los filtros convolucionales detectan patrones en imágenes usando relaciones espaciales, podemos detectar patrones financieros usando relaciones económicas.

## Conceptos Fundamentales

### Redes Financieras vs Redes Tradicionales

**Diferencias Clave:**

| Aspecto | Redes de Imágenes | Redes Financieras |
|---------|-------------------|-------------------|
| **Nodos** | Píxeles | Empresas/Activos |
| **Conexiones** | Proximidad espacial | Relaciones económicas |
| **Distancia** | Euclidiana | Correlación/Causalidad |
| **Estabilidad** | Estática | Dinámica en el tiempo |

### Construcción de Redes Financieras

```python
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FinancialNetworkBuilder:
    def __init__(self, method='correlation'):
        self.method = method
        self.threshold = 0.3
        
    def build_correlation_network(self, returns_df):
        """
        Construye red basada en correlaciones de rendimientos
        """
        # Calcular matriz de correlación
        corr_matrix = returns_df.corr()
        
        # Crear grafo
        G = nx.Graph()
        
        # Agregar nodos (empresas)
        for company in returns_df.columns:
            G.add_node(company)
        
        # Agregar aristas basadas en correlación
        for i, company1 in enumerate(returns_df.columns):
            for j, company2 in enumerate(returns_df.columns[i+1:], i+1):
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) > self.threshold:
                    G.add_edge(company1, company2, 
                              weight=abs(correlation),
                              correlation=correlation)
        
        return G
    
    def build_news_network(self, news_mentions):
        """
        Construye red basada en co-menciones en noticias
        """
        G = nx.Graph()
        
        # Procesar artículos de noticias
        for article in news_mentions:
            companies_mentioned = article['companies']
            
            # Crear conexiones entre empresas mencionadas juntas
            for i, company1 in enumerate(companies_mentioned):
                for company2 in companies_mentioned[i+1:]:
                    if G.has_edge(company1, company2):
                        G[company1][company2]['weight'] += 1
                    else:
                        G.add_edge(company1, company2, weight=1)
        
        return G
    
    def build_supply_chain_network(self, supply_relationships):
        """
        Construye red basada en relaciones de cadena de suministro
        """
        G = nx.DiGraph()  # Dirigido para supplier -> customer
        
        for relationship in supply_relationships:
            supplier = relationship['supplier']
            customer = relationship['customer']
            importance = relationship['revenue_percentage']
            
            G.add_edge(supplier, customer, 
                      weight=importance,
                      relationship_type='supply_chain')
        
        return G
```

## Graph Neural Networks para Finanzas

### Arquitectura Base

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class FinancialGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, network_type='gcn'):
        super().__init__()
        self.network_type = network_type
        
        if network_type == 'gcn':
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif network_type == 'gat':
            self.conv1 = GATConv(num_features, hidden_dim, heads=8, dropout=0.1)
            self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=1, dropout=0.1)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Primer layer de convolución
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Segundo layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Clasificación/Predicción
        out = self.classifier(x)
        
        return out
```

### Aplicación: Predicción de Riesgo Sectorial

```python
class SectorRiskPredictor:
    def __init__(self, gnn_model):
        self.model = gnn_model
        
    def prepare_sector_network(self, company_data, sector_classifications):
        """
        Prepara red para análisis de riesgo sectorial
        """
        # Features por empresa
        features = []
        node_mapping = {}
        
        for i, (company, data) in enumerate(company_data.items()):
            node_mapping[company] = i
            
            company_features = [
                data['market_cap_log'],
                data['pe_ratio'],
                data['debt_to_equity'],
                data['beta'],
                data['roa'],
                data['current_ratio'],
                data['revenue_growth']
            ]
            features.append(company_features)
        
        # Crear edges basados en sector y correlaciones
        edge_index = []
        edge_attr = []
        
        # Intra-sector connections
        for sector, companies in sector_classifications.items():
            for i, company1 in enumerate(companies):
                for company2 in companies[i+1:]:
                    if company1 in node_mapping and company2 in node_mapping:
                        idx1, idx2 = node_mapping[company1], node_mapping[company2]
                        
                        edge_index.extend([[idx1, idx2], [idx2, idx1]])
                        
                        # Edge attributes: sector similarity, correlation
                        correlation = self.calculate_correlation(company1, company2)
                        edge_attr.extend([[1.0, correlation], [1.0, correlation]])
        
        # Inter-sector connections (supply chain, etc.)
        for connection in self.supply_chain_connections:
            if connection['supplier'] in node_mapping and connection['customer'] in node_mapping:
                idx1 = node_mapping[connection['supplier']]
                idx2 = node_mapping[connection['customer']]
                
                edge_index.extend([[idx1, idx2], [idx2, idx1]])
                edge_attr.extend([[0.5, connection['strength']], [0.5, connection['strength']]])
        
        return torch.tensor(features), torch.tensor(edge_index).t(), torch.tensor(edge_attr)
    
    def predict_contagion_risk(self, shock_companies, network_data):
        """
        Predice propagación de riesgo a través de la red
        """
        features, edge_index, edge_attr = network_data
        
        # Inicializar shock
        shock_vector = torch.zeros(len(features))
        for company in shock_companies:
            if company in self.node_mapping:
                shock_vector[self.node_mapping[company]] = 1.0
        
        # Simular propagación
        propagation_steps = []
        current_shock = shock_vector.clone()
        
        for step in range(10):  # 10 pasos de propagación
            # Aplicar GNN para predecir próximo estado
            model_input = torch.cat([features, current_shock.unsqueeze(1)], dim=1)
            next_shock = self.model(model_input, edge_index)
            
            propagation_steps.append(next_shock.detach().clone())
            current_shock = next_shock.squeeze()
        
        return propagation_steps
```

## Casos de Estudio

### Caso 1: Crisis Financiera 2008 - Análisis de Red

```python
def analyze_2008_crisis_network():
    """
    Analiza la evolución de la red durante la crisis de 2008
    """
    # Períodos de análisis
    periods = {
        'pre_crisis': ('2006-01-01', '2007-06-30'),
        'crisis_onset': ('2007-07-01', '2008-03-31'),
        'peak_crisis': ('2008-04-01', '2008-12-31'),
        'recovery': ('2009-01-01', '2009-12-31')
    }
    
    network_evolution = {}
    
    for period_name, (start_date, end_date) in periods.items():
        # Datos del período
        period_returns = get_returns_data(start_date, end_date)
        
        # Construir red
        network_builder = FinancialNetworkBuilder()
        G = network_builder.build_correlation_network(period_returns)
        
        # Métricas de red
        metrics = {
            'density': nx.density(G),
            'clustering': nx.average_clustering(G),
            'path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
            'centrality_concentration': calculate_centrality_concentration(G),
            'modularity': calculate_modularity_by_sector(G)
        }
        
        network_evolution[period_name] = {
            'graph': G,
            'metrics': metrics,
            'central_nodes': identify_central_nodes(G)
        }
    
    return network_evolution

def analyze_crisis_propagation():
    """
    Análisis específico de propagación durante la crisis
    """
    # Red pre-crisis vs crisis
    pre_crisis_network = build_network('2006-01-01', '2007-06-30')
    crisis_network = build_network('2008-01-01', '2008-12-31')
    
    findings = {
        'density_change': nx.density(crisis_network) / nx.density(pre_crisis_network),
        'lehman_centrality': {
            'pre_crisis': nx.betweenness_centrality(pre_crisis_network)['LEH'],
            'during_crisis': nx.betweenness_centrality(crisis_network)['LEH'] if 'LEH' in crisis_network else 0
        },
        'financial_sector_clustering': analyze_sector_clustering(crisis_network, 'Financial'),
        'contagion_paths': find_contagion_paths(crisis_network, source='LEH')
    }
    
    return findings
```

**Hallazgos Clave:**
- La red se volvió **altamente conectada** durante la crisis
- **Lehman Brothers** emergió como nodo central
- La **modularidad sectorial** desapareció (todos los sectores se correlacionaron)
- **Paths de contagio** fueron principalmente a través del sector financiero

### Caso 2: Análisis de Sentiment en Redes

```python
class SentimentNetworkAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    def build_sentiment_network(self, news_data, companies):
        """
        Construye red basada en sentiment de noticias
        """
        G = nx.Graph()
        
        # Procesar noticias por empresa
        company_sentiments = {}
        
        for article in news_data:
            mentioned_companies = self.extract_companies(article['text'], companies)
            article_sentiment = self.sentiment_analyzer(article['text'])[0]
            
            sentiment_score = article_sentiment['score'] if article_sentiment['label'] == 'POSITIVE' else -article_sentiment['score']
            
            # Agregar sentiment a empresas mencionadas
            for company in mentioned_companies:
                if company not in company_sentiments:
                    company_sentiments[company] = []
                company_sentiments[company].append(sentiment_score)
        
        # Crear conexiones basadas en co-ocurrencia de sentiment
        for article in news_data:
            mentioned_companies = self.extract_companies(article['text'], companies)
            if len(mentioned_companies) > 1:
                article_sentiment = self.get_article_sentiment(article['text'])
                
                for i, company1 in enumerate(mentioned_companies):
                    for company2 in mentioned_companies[i+1:]:
                        if G.has_edge(company1, company2):
                            # Actualizar peso basado en sentiment compartido
                            G[company1][company2]['sentiment_correlation'] += article_sentiment
                            G[company1][company2]['co_mentions'] += 1
                        else:
                            G.add_edge(company1, company2, 
                                     sentiment_correlation=article_sentiment,
                                     co_mentions=1)
        
        return G
    
    def predict_sentiment_contagion(self, network, initial_sentiment_shock):
        """
        Predice propagación de sentiment a través de la red
        """
        # Modelo de difusión de sentiment
        sentiment_states = {node: 0 for node in network.nodes()}
        
        # Aplicar shock inicial
        for company, sentiment in initial_sentiment_shock.items():
            if company in sentiment_states:
                sentiment_states[company] = sentiment
        
        # Simular propagación
        for iteration in range(10):
            new_states = sentiment_states.copy()
            
            for node in network.nodes():
                neighbor_influence = 0
                total_weight = 0
                
                for neighbor in network.neighbors(node):
                    edge_data = network[node][neighbor]
                    weight = edge_data.get('sentiment_correlation', 0)
                    
                    neighbor_influence += weight * sentiment_states[neighbor]
                    total_weight += abs(weight)
                
                if total_weight > 0:
                    # Combinar sentiment propio con influencia de vecinos
                    new_states[node] = 0.7 * sentiment_states[node] + 0.3 * (neighbor_influence / total_weight)
        
        return new_states
```

## Modelado de Riesgo con Redes

### Dynamic Risk Networks

```python
class DynamicRiskNetwork:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.risk_networks = {}
        
    def build_rolling_risk_networks(self, returns_data):
        """
        Construye redes de riesgo con ventanas móviles
        """
        for i in range(self.window_size, len(returns_data)):
            window_data = returns_data.iloc[i-self.window_size:i]
            date = returns_data.index[i]
            
            # Construir red de correlaciones
            corr_network = self.build_correlation_network(window_data)
            
            # Calcular métricas de riesgo sistémico
            systemic_risk_metrics = self.calculate_systemic_risk(corr_network, window_data)
            
            self.risk_networks[date] = {
                'network': corr_network,
                'systemic_risk': systemic_risk_metrics,
                'network_density': nx.density(corr_network),
                'avg_clustering': nx.average_clustering(corr_network)
            }
    
    def calculate_systemic_risk(self, network, returns_data):
        """
        Calcula métricas de riesgo sistémico
        """
        # 1. Network density como proxy de contagio
        density = nx.density(network)
        
        # 2. Concentration de centralidad
        centrality = nx.eigenvector_centrality(network)
        centrality_concentration = np.std(list(centrality.values()))
        
        # 3. CoVaR (Conditional Value at Risk)
        covar_matrix = self.calculate_covar_matrix(returns_data)
        
        # 4. Network CoVaR
        network_covar = self.calculate_network_covar(network, covar_matrix)
        
        return {
            'network_density': density,
            'centrality_concentration': centrality_concentration,
            'average_covar': np.mean(covar_matrix),
            'network_covar': network_covar,
            'systemic_risk_score': self.aggregate_risk_score(density, centrality_concentration, network_covar)
        }
    
    def calculate_covar_matrix(self, returns_data, quantile=0.05):
        """
        Calcula Conditional Value at Risk entre pares de activos
        """
        companies = returns_data.columns
        covar_matrix = np.zeros((len(companies), len(companies)))
        
        for i, company1 in enumerate(companies):
            for j, company2 in enumerate(companies):
                if i != j:
                    # VaR de company2
                    var_company2 = np.quantile(returns_data[company2], quantile)
                    
                    # Rendimientos de company1 cuando company2 está en stress
                    stress_condition = returns_data[company2] <= var_company2
                    conditional_returns = returns_data[company1][stress_condition]
                    
                    # CoVaR
                    if len(conditional_returns) > 0:
                        covar = np.quantile(conditional_returns, quantile)
                        covar_matrix[i, j] = covar
        
        return covar_matrix
```

### Stress Testing con Redes

```python
class NetworkStressTester:
    def __init__(self, risk_network):
        self.network = risk_network
        
    def node_removal_stress_test(self, removal_strategy='centrality'):
        """
        Test de estrés removiendo nodos críticos
        """
        original_network = self.network.copy()
        stress_results = {}
        
        if removal_strategy == 'centrality':
            # Remover nodos por orden de centralidad
            centrality = nx.betweenness_centrality(original_network)
            nodes_to_remove = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)
        elif removal_strategy == 'random':
            nodes_to_remove = list(original_network.nodes())
            np.random.shuffle(nodes_to_remove)
        
        for i, node in enumerate(nodes_to_remove[:10]):  # Top 10 nodos
            test_network = original_network.copy()
            test_network.remove_node(node)
            
            # Métricas post-remoción
            if nx.is_connected(test_network):
                avg_path_length = nx.average_shortest_path_length(test_network)
                efficiency = nx.global_efficiency(test_network)
            else:
                # Red fragmentada
                largest_component = max(nx.connected_components(test_network), key=len)
                subgraph = test_network.subgraph(largest_component)
                avg_path_length = nx.average_shortest_path_length(subgraph)
                efficiency = len(largest_component) / len(original_network)
            
            stress_results[node] = {
                'removed_order': i + 1,
                'avg_path_length': avg_path_length,
                'network_efficiency': efficiency,
                'largest_component_size': len(largest_component) if not nx.is_connected(test_network) else len(test_network),
                'fragmentation_impact': 1 - (len(largest_component) / len(original_network)) if not nx.is_connected(test_network) else 0
            }
        
        return stress_results
    
    def cascading_failure_simulation(self, initial_shock_nodes, failure_threshold=0.3):
        """
        Simula fallos en cascada en la red
        """
        network = self.network.copy()
        failed_nodes = set(initial_shock_nodes)
        cascade_steps = []
        
        step = 0
        while True:
            step += 1
            new_failures = set()
            
            # Calcular estrés en nodos restantes
            for node in network.nodes():
                if node not in failed_nodes:
                    # Contar vecinos fallados
                    neighbors = list(network.neighbors(node))
                    failed_neighbors = len([n for n in neighbors if n in failed_nodes])
                    
                    # Ratio de vecinos fallados
                    if len(neighbors) > 0:
                        failure_ratio = failed_neighbors / len(neighbors)
                        
                        if failure_ratio >= failure_threshold:
                            new_failures.add(node)
            
            if not new_failures:
                break  # No more cascading failures
                
            failed_nodes.update(new_failures)
            cascade_steps.append({
                'step': step,
                'new_failures': list(new_failures),
                'total_failed': len(failed_nodes),
                'failure_percentage': len(failed_nodes) / len(self.network.nodes())
            })
            
            # Remover nodos fallados para próxima iteración
            network.remove_nodes_from(new_failures)
        
        return {
            'total_steps': step,
            'cascade_progression': cascade_steps,
            'final_failure_rate': len(failed_nodes) / len(self.network.nodes()),
            'surviving_nodes': len(self.network.nodes()) - len(failed_nodes)
        }
```

## Aplicaciones Prácticas

### 1. Portfolio Construction

```python
class NetworkAwarePortfolio:
    def __init__(self, returns_data, risk_network):
        self.returns_data = returns_data
        self.network = risk_network
        
    def optimize_network_diversification(self, target_return=0.10):
        """
        Optimización de portfolio considerando estructura de red
        """
        from scipy.optimize import minimize
        
        n_assets = len(self.returns_data.columns)
        expected_returns = self.returns_data.mean()
        cov_matrix = self.returns_data.cov()
        
        # Función objetivo: minimizar riesgo + penalización por concentración de red
        def objective(weights):
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            network_concentration_penalty = self.calculate_network_concentration(weights)
            
            return portfolio_var + 0.1 * network_concentration_penalty
        
        # Restricciones
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Suma = 1
            {'type': 'ineq', 'fun': lambda x: np.dot(expected_returns, x) - target_return}  # Return target
        ]
        
        bounds = [(0, 0.1) for _ in range(n_assets)]  # Max 10% per asset
        
        # Optimización
        result = minimize(
            objective,
            x0=np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def calculate_network_concentration(self, weights):
        """
        Penaliza concentración en nodos centrales de la red
        """
        centrality = nx.eigenvector_centrality(self.network)
        companies = self.returns_data.columns
        
        concentration_penalty = 0
        for i, company in enumerate(companies):
            if company in centrality:
                # Penalizar más peso en nodos más centrales
                concentration_penalty += weights[i] * centrality[company]
        
        return concentration_penalty
```

### 2. Risk Management

```python
class NetworkRiskManager:
    def __init__(self, portfolio_weights, risk_network):
        self.weights = portfolio_weights
        self.network = risk_network
        
    def calculate_network_var(self, confidence_level=0.95, holding_period=1):
        """
        Calcula VaR considerando efectos de red
        """
        # VaR tradicional
        portfolio_returns = self.calculate_portfolio_returns()
        traditional_var = np.quantile(portfolio_returns, 1 - confidence_level)
        
        # Ajuste por concentración de red
        network_concentration = self.calculate_portfolio_network_exposure()
        concentration_multiplier = 1 + 0.5 * network_concentration  # Hasta 50% de aumento
        
        # VaR ajustado por red
        network_adjusted_var = traditional_var * concentration_multiplier
        
        return {
            'traditional_var': traditional_var,
            'network_adjusted_var': network_adjusted_var,
            'network_concentration': network_concentration,
            'concentration_multiplier': concentration_multiplier
        }
    
    def real_time_contagion_monitoring(self):
        """
        Monitoreo en tiempo real de riesgo de contagio
        """
        # Identificar posiciones en nodos críticos
        centrality = nx.betweenness_centrality(self.network)
        
        critical_exposures = {}
        for i, company in enumerate(self.portfolio.companies):
            if company in centrality and self.weights[i] > 0.05:  # >5% weight
                critical_exposures[company] = {
                    'weight': self.weights[i],
                    'centrality': centrality[company],
                    'risk_score': self.weights[i] * centrality[company]
                }
        
        # Alert si hay concentración excesiva
        total_critical_exposure = sum([exp['weight'] for exp in critical_exposures.values()])
        
        return {
            'critical_exposures': critical_exposures,
            'total_critical_weight': total_critical_exposure,
            'alert_level': 'HIGH' if total_critical_exposure > 0.3 else 'MEDIUM' if total_critical_exposure > 0.2 else 'LOW'
        }
```

## Limitaciones y Consideraciones

### Desafíos Técnicos

**1. Calidad de Datos:**
```python
def validate_network_data_quality(network_data):
    """
    Validación de calidad para datos de red
    """
    quality_checks = {
        'missing_nodes': check_missing_company_data(network_data),
        'edge_stability': check_edge_temporal_stability(network_data),
        'spurious_correlations': detect_spurious_connections(network_data),
        'data_freshness': check_data_recency(network_data)
    }
    
    return quality_checks
```

**2. Computational Complexity:**
- Redes grandes (>1000 nodos) requieren optimizaciones
- Actualización en tiempo real challenging
- Trade-off entre precisión y velocidad

**3. Model Overfitting:**
- Riesgo de overfitting a patrones específicos de red
- Necesidad de validación cruzada temporal
- Robustez a cambios estructurales

### Mejores Prácticas

**1. Network Construction:**
```python
def robust_network_construction():
    """
    Mejores prácticas para construcción de redes robustas
    """
    best_practices = {
        'multiple_data_sources': 'Combinar correlaciones, noticias, fundamentales',
        'dynamic_thresholds': 'Ajustar thresholds basado en régimen de mercado',
        'temporal_validation': 'Validar estabilidad de conexiones en el tiempo',
        'sector_awareness': 'Considerar estructura sectorial conocida',
        'outlier_handling': 'Detectar y manejar períodos anómalos'
    }
    return best_practices
```

**2. Risk Management:**
- **Límites de concentración** en nodos centrales
- **Monitoreo continuo** de métricas de red
- **Stress testing** regular con escenarios de contagio
- **Diversificación explícita** por estructura de red

---

*El análisis de redes ofrece una perspectiva única sobre la estructura y dinámicas de los mercados financieros. Al considerar no solo los activos individuales sino también sus interconexiones, podemos desarrollar estrategias más robustas y sistemas de gestión de riesgo más efectivos.*