# Guía de Seguridad - Infraestructura Trading

## ⚠️ IMPORTANTE: Configuración de Seguridad

Este directorio contiene templates de infraestructura que **REQUIEREN configuración de seguridad** antes del uso en producción.

## 🔐 Archivos que DEBES Personalizar

### 1. Variables de Entorno
```bash
# Copiar template de ejemplo
cp .env.example .env

# Editar con tus credenciales reales
nano .env
```

**NUNCA commits archivos .env reales al repositorio**

### 2. Passwords y Secrets
Cambiar TODOS los passwords por defecto en:

- **PostgreSQL**: `POSTGRES_PASSWORD`
- **Redis**: `requirepass` en `redis.conf`
- **MinIO**: `MINIO_ROOT_USER` y `MINIO_ROOT_PASSWORD`
- **Grafana**: `GRAFANA_PASSWORD`

### 3. API Keys
Configurar tus propias API keys:

- **Polygon.io**: Para datos de mercado
- **Alpha Vantage**: Datos financieros alternativos
- **IEX Cloud**: Datos adicionales
- **Telegram Bot**: Para alertas

### 4. Kubernetes Secrets
Los secrets en Kubernetes están en **base64** pero NO son encriptación:

```bash
# Generar nuevos secrets
echo -n "tu_password_real" | base64

# Actualizar en postgres.yaml y trading-engine.yaml
```

## 🛡️ Mejores Prácticas de Seguridad

### Producción
1. **Usar un vault de secrets** (HashiCorp Vault, AWS Secrets Manager)
2. **Habilitar TLS/SSL** en todas las comunicaciones
3. **Configurar firewalls** y network policies
4. **Auditar accesos** y logs de seguridad
5. **Rotar passwords** periódicamente

### Desarrollo
1. **Usar passwords únicos** (no reutilizar)
2. **Mantener .env fuera del repo**
3. **Usar paper trading** en desarrollo
4. **Limitar acceso a red** (VPN/firewall)

## 🚨 Datos que NUNCA Debes Commitear

- ❌ Passwords reales
- ❌ API keys reales
- ❌ Certificados privados
- ❌ Archivos .env con datos reales
- ❌ Backups de base de datos
- ❌ Logs que contengan credenciales

## ✅ Lo que SÍ es Seguro Commitear

- ✅ Templates (.env.example)
- ✅ Configuraciones con placeholders
- ✅ Scripts de setup
- ✅ Documentación
- ✅ Configuraciones de Kubernetes (sin secrets reales)

## 🔧 Setup Rápido y Seguro

```bash
# 1. Clonar configuraciones
git clone <repo>
cd infrastructure/

# 2. Crear archivo de entorno
cp .env.example .env

# 3. Generar passwords seguros
openssl rand -base64 32  # Para cada password

# 4. Editar .env con valores reales
vim .env

# 5. Verificar que .env está en .gitignore
git status  # No debe mostrar .env

# 6. Deploy seguro
docker-compose up -d
```

## 📞 Contacto para Problemas de Seguridad

Si encuentras vulnerabilidades de seguridad, repórtalas de forma responsable:

1. **NO** abras issues públicos
2. Contacta directamente al mantenedor
3. Proporciona detalles del problema
4. Espera confirmación antes de disclosure público

## 🔄 Rotación de Credenciales

### Frecuencia Recomendada
- **Passwords de DB**: Cada 90 días
- **API Keys**: Según política del proveedor
- **Certificates**: Antes de expiración
- **Tokens de acceso**: Cada 30 días

### Proceso de Rotación
1. Generar nuevas credenciales
2. Actualizar en vault/config
3. Restart servicios
4. Revocar credenciales antiguas
5. Verificar funcionamiento