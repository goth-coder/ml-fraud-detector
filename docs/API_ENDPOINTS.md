# 📡 API Endpoints - Dashboard de Detecção de Fraudes

Documentação completa dos endpoints REST do backend Flask.

---

## 🏥 Health Check

### `GET /health`

Verifica status da aplicação e versão do modelo.

**Response:**
```json
{
  "status": "healthy",
  "model_version": "v2.1.0"
}
```

---

## 🎲 Simulação de Transações

### `POST /api/simulate`

Simula uma transação (legítima ou fraudulenta), classifica com o modelo e salva no banco.

**Request Body:**
```json
{
  "transaction_type": "legitimate" | "fraud"
}
```

**Response Success (200):**
```json
{
  "success": true,
  "transaction_id": 123,
  "classification_id": 456,
  "is_fraud": false,
  "fraud_probability": 0.0023,
  "fraud_probability_percent": "0.23%",
  "confidence": "Baixa",
  "model_version": "v2.1.0",
  "predicted_at": "2025-10-06T10:30:00.123456",
  "transaction_type": "legitimate"
}
```

**Response Error (400):**
```json
{
  "success": false,
  "error": "Campo \"transaction_type\" é obrigatório (body JSON: {\"transaction_type\": \"legitimate\" | \"fraud\"})"
}
```

**Response Error (500):**
```json
{
  "success": false,
  "error": "Mensagem de erro detalhada"
}
```

**Workflow Interno:**
1. Transaction Generator → busca transação real do PostgreSQL (test_data)
2. Model Service → classifica com XGBoost v2.1.0
3. Database Service → salva em `classification_results` e `simulated_transactions`
4. Retorna resultado completo com IDs e metadados

**Validações:**
- ✅ `transaction_type` obrigatório
- ✅ Valores permitidos: `"legitimate"` ou `"fraud"`
- ✅ Content-Type: `application/json`

---

## 📊 Estatísticas Agregadas

### `GET /api/stats`

Retorna estatísticas de classificações em um período.

**Query Parameters:**
- `hours` (opcional): Período em horas (padrão: 24, min: 1, max: 168)

**Example:**
```
GET /api/stats?hours=24
```

**Response Success (200):**
```json
{
  "success": true,
  "stats": {
    "total": 150,
    "fraud_count": 30,
    "fraud_percentage": 20.0,
    "avg_probability": 0.4523,
    "max_probability": 0.9987,
    "min_probability": 0.0001,
    "by_hour": [
      {
        "hour": "2025-10-06T10:00:00",
        "count": 15
      },
      {
        "hour": "2025-10-06T11:00:00",
        "count": 23
      }
    ],
    "period_hours": 24
  }
}
```

**Response Error (400):**
```json
{
  "success": false,
  "error": "hours deve estar entre 1 e 168"
}
```

**Métricas:**
- `total`: Total de classificações no período
- `fraud_count`: Quantidade de fraudes detectadas
- `fraud_percentage`: Percentual de fraudes (%)
- `avg_probability`: Probabilidade média de fraude
- `max_probability`: Maior probabilidade detectada
- `min_probability`: Menor probabilidade detectada
- `by_hour`: Array com contagem por hora
- `period_hours`: Período analisado em horas

**Validações:**
- ✅ `hours` deve estar entre 1 e 168 (7 dias)

---

## 📜 Histórico de Classificações

### `GET /api/history`

Retorna histórico das últimas classificações ordenadas por timestamp decrescente.

**Query Parameters:**
- `limit` (opcional): Número máximo de registros (padrão: 50, min: 1, max: 1000)

**Example:**
```
GET /api/history?limit=2
```

**Response Success (200):**
```json
{
  "success": true,
  "count": 2,
  "history": [
    {
      "id": 456,
      "predicted_at": "2025-10-06T10:30:00.123456",
      "is_fraud": true,
      "fraud_probability": 0.9876,
      "model_version": "v2.1.0",
      "source": "webapp"
    },
    {
      "id": 455,
      "predicted_at": "2025-10-06T10:29:45.987654",
      "is_fraud": false,
      "fraud_probability": 0.0012,
      "model_version": "v2.1.0",
      "source": "webapp"
    }
  ]
}
```

**Response Error (400):**
```json
{
  "success": false,
  "error": "limit deve estar entre 1 e 1000"
}
```

**Campos:**
- `id`: ID da classificação no banco
- `predicted_at`: Timestamp da predição (ISO 8601)
- `is_fraud`: Boolean indicando se é fraude
- `fraud_probability`: Probabilidade de fraude (0.0 a 1.0)
- `model_version`: Versão do modelo usado
- `source`: Origem da predição (`webapp`, `api`, `batch`)

**Validações:**
- ✅ `limit` deve estar entre 1 e 1000

---

## 🔐 Segurança e CORS

- **CORS habilitado** para desenvolvimento (desabilitar em produção)
- **Content-Type**: `application/json` obrigatório em POSTs
- **Logging**: Todos os erros são logados com `logger.error()`

---

## 🧪 Testes com cURL

### Simular transação legítima:
```bash
curl -X POST http://localhost:5000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"transaction_type": "legitimate"}'
```

### Simular transação fraudulenta:
```bash
curl -X POST http://localhost:5000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"transaction_type": "fraud"}'
```

### Obter estatísticas (últimas 24h):
```bash
curl http://localhost:5000/api/stats?hours=24
```

### Obter histórico (últimas 50):
```bash
curl http://localhost:5000/api/history?limit=50
```

### Health check:
```bash
curl http://localhost:5000/health
```

---

## 📝 Decisões Técnicas

### Por que não tem `/api/predict`?

**Removido** porque o usuário do webapp **não digita features manualmente** (V1-V28).

- ✅ **Frontend**: Apenas botões "Simular Legítima" e "Simular Fraude"
- ✅ **Backend**: Gera transação real do PostgreSQL automaticamente
- ✅ **UX**: Mais simples e intuitivo

**Caso futuro:** Se precisar de endpoint para sistemas externos enviarem features, pode ser adicionado posteriormente.

---

## 🚀 Próximos Passos

- [ ] Frontend Dashboard (HTML/CSS/JS)
- [ ] WebSocket para streaming real-time (opcional)
- [ ] Autenticação (se necessário)
- [ ] Rate limiting (produção)
- [ ] Documentação Swagger/OpenAPI (opcional)
