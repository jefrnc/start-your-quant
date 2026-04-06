> 🇪🇸 [Leer en Español](SECURITY.es.md) | 🇺🇸 **English**

# Security Guide - Trading Infrastructure

## IMPORTANT: Security Configuration

This directory contains infrastructure templates that **REQUIRE security configuration** before use in production.

## Files You MUST Customize

### 1. Environment Variables
```bash
# Copy the example template
cp .env.example .env

# Edit with your real credentials
nano .env
```

**NEVER commit real .env files to the repository**

### 2. Passwords and Secrets
Change ALL default passwords in:

- **PostgreSQL**: `POSTGRES_PASSWORD`
- **Redis**: `requirepass` in `redis.conf`
- **MinIO**: `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`
- **Grafana**: `GRAFANA_PASSWORD`

### 3. API Keys
Configure your own API keys:

- **Polygon.io**: For market data
- **Alpha Vantage**: Alternative financial data
- **IEX Cloud**: Additional data
- **Telegram Bot**: For alerts

### 4. Kubernetes Secrets
Secrets in Kubernetes are in **base64** but are NOT encryption:

```bash
# Generate new secrets
echo -n "your_real_password" | base64

# Update in postgres.yaml and trading-engine.yaml
```

## Security Best Practices

### Production
1. **Use a secrets vault** (HashiCorp Vault, AWS Secrets Manager)
2. **Enable TLS/SSL** on all communications
3. **Configure firewalls** and network policies
4. **Audit access** and security logs
5. **Rotate passwords** periodically

### Development
1. **Use unique passwords** (don't reuse)
2. **Keep .env out of the repo**
3. **Use paper trading** in development
4. **Limit network access** (VPN/firewall)

## Data You Should NEVER Commit

- Real passwords
- Real API keys
- Private certificates
- .env files with real data
- Database backups
- Logs containing credentials

## What IS Safe to Commit

- Templates (.env.example)
- Configurations with placeholders
- Setup scripts
- Documentation
- Kubernetes configurations (without real secrets)

## Quick and Secure Setup

```bash
# 1. Clone configurations
git clone <repo>
cd infrastructure/

# 2. Create environment file
cp .env.example .env

# 3. Generate secure passwords
openssl rand -base64 32  # For each password

# 4. Edit .env with real values
vim .env

# 5. Verify .env is in .gitignore
git status  # Should not show .env

# 6. Secure deploy
docker-compose up -d
```

## Contact for Security Issues

If you find security vulnerabilities, report them responsibly:

1. **DO NOT** open public issues
2. Contact the maintainer directly
3. Provide details of the issue
4. Wait for confirmation before public disclosure

## Credential Rotation

### Recommended Frequency
- **DB Passwords**: Every 90 days
- **API Keys**: Per provider policy
- **Certificates**: Before expiration
- **Access Tokens**: Every 30 days

### Rotation Process
1. Generate new credentials
2. Update in vault/config
3. Restart services
4. Revoke old credentials
5. Verify functionality
