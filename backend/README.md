# CapFlex Backend — Microservices

Arquitectura de dos microservicios FastAPI orquestados con Docker Compose.

| Servicio        | Puerto | Responsabilidad                        |
|-----------------|--------|----------------------------------------|
| embedding-svc   | 8001   | Generación de embeddings CLIP          |
| clustering-svc  | 8002   | Clustering CapFlex (tabular/embeddings)|

---

## Requisitos

- Docker + Docker Compose
- NVIDIA GPU + nvidia-container-toolkit (opcional, para embedding-svc)

---

## Levantar los servicios

```bash
docker-compose up --build
```

---

## Flujos de uso

### 1. Clustering con CSV tabular

```bash
# Lanzar job
curl -X POST http://localhost:8002/clustering/run \
  -F "file=@iris.csv" \
  -F "input_type=tabular" \
  -F "label_column=Species" \
  -F "target_cardinality=50,50,50" \
  -F "delta=0.1"

# Respuesta: {"job_id": "abc-123", "status": "pending", ...}

# Consultar estado
curl http://localhost:8002/clustering/status/abc-123

# Obtener métricas y Pareto front
curl http://localhost:8002/clustering/results/abc-123

# Descargar CSV con asignaciones
curl -O http://localhost:8002/clustering/download/abc-123
```

---

### 2. Clustering con imágenes (generando embeddings)

```bash
# Paso 1 — Generar embeddings (embedding-svc)
curl -X POST http://localhost:8001/embeddings/images \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "files=@img3.jpg"

# Respuesta: {"job_id": "emb-456", "status": "pending", ...}

# Consultar estado del embedding
curl http://localhost:8001/embeddings/status/emb-456

# Paso 2 — Clustering usando el job_id del embedding
curl -X POST http://localhost:8002/clustering/run \
  -F "embedding_job_id=emb-456" \
  -F "target_cardinality=10,10,10" \
  -F "delta=0.1"
```

---

### 3. Clustering con CSV de embeddings propios

```bash
# El usuario ya tiene su embeddings.csv con columnas emb_0, emb_1, ..., emb_511
curl -X POST http://localhost:8002/clustering/run \
  -F "file=@embeddings.csv" \
  -F "input_type=embeddings" \
  -F "embedding_prefix=emb" \
  -F "target_cardinality=16,17" \
  -F "delta=0.05"
```

---

### 4. Clustering con texto (generando embeddings)

```bash
# CSV con columna de texto
curl -X POST http://localhost:8001/embeddings/texts \
  -F "file=@reviews.csv" \
  -F "text_column=review_text" \
  -F "id_column=product_id"

# Respuesta: {"job_id": "txt-789", ...}

# Luego clustering con el job_id
curl -X POST http://localhost:8002/clustering/run \
  -F "embedding_job_id=txt-789" \
  -F "target_cardinality=30,30,30" \
  -F "delta=0.1"
```

---

## Documentación interactiva (Swagger)

- Embedding service: http://localhost:8001/docs
- Clustering service: http://localhost:8002/docs

---

## Estructura del proyecto

```
capflex-backend/
├── docker-compose.yml
├── shared_data/              ← volumen compartido entre servicios
│   ├── uploads/              ← CSVs e imágenes subidas
│   ├── embeddings/           ← CSVs de embeddings generados
│   └── results/              ← CSVs de clustering con asignaciones
├── embedding-svc/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   └── clip_embedder.py
└── clustering-svc/
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py
    └── capflex.py
```

## Parámetros del endpoint /clustering/run

| Parámetro            | Tipo    | Descripción                                              |
|----------------------|---------|----------------------------------------------------------|
| `file`               | File    | CSV tabular o CSV de embeddings                          |
| `embedding_job_id`   | string  | job_id de embedding-svc (alternativa a file)            |
| `input_type`         | string  | `tabular` o `embeddings`                                 |
| `embedding_prefix`   | string  | Prefijo de columnas de embeddings (default: `emb`)       |
| `label_column`       | string  | Columna de etiquetas reales (opcional, para calcular AMI)|
| `target_cardinality` | string  | Tamaños objetivo separados por coma: `50,50,50`          |
| `delta`              | float   | Tolerancia de cardinalidad (0.0 – 1.0)                   |
| `max_iter`           | int     | Máximo de combinaciones a explorar (None = automático)   |
