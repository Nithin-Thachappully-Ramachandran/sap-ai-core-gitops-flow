# AquaVerde AI Core Docker Fix

## What was fixed

1. `model.joblib` is exported as:

```python
{"model": model, "features": FEATURES, "horizon": 7}
```

The previous `serve_aicore.py` expected:

```python
{"models": ..., "features": ..., "horizon": ...}
```

That caused:

```text
KeyError: 'models'
```

2. LightGBM needs the Linux OpenMP runtime in slim Docker images:

```text
libgomp.so.1
```

The Dockerfile now installs `libgomp1`.

3. MLflow/cloudpickle model export needs runtime dependencies:

```text
mlflow
cloudpickle
```

4. `/v2/healthz` is now a lightweight health endpoint. Use `/v2/readyz` to validate model loading.

## Files to replace in your folder

Copy these files into:

```text
C:\Users\de34670\Downloads\CPG_DEMO\aquaverde-quantile-deploy
```

Replace the existing versions:

- `Dockerfile.aicore`
- `requirements-aicore.txt`
- `serve_aicore.py`

Keep your downloaded files:

- `model.joblib`
- `latest_features.csv`

## Build

```powershell
cd C:\Users\de34670\Downloads\CPG_DEMO\aquaverde-quantile-deploy

docker build --no-cache --platform linux/amd64 -t aquaverde-inference:3.3 -f Dockerfile.aicore .
```

## Run

```powershell
docker run --rm -p 8080:8080 aquaverde-inference:3.3
```

## Test in a second PowerShell

```powershell
Invoke-RestMethod -Uri http://localhost:8080/v2/healthz
```

Expected:

```json
{"status":"ok"}
```

Then check model readiness:

```powershell
Invoke-RestMethod -Uri http://localhost:8080/v2/readyz
```

Expected shape:

```json
{
  "status": "ready",
  "skus_loaded": 1000,
  "horizon_days": 7,
  "bundle_keys": ["model", "features", "horizon"]
}
```

Then inference:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:8080/v2/replenish `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"sku":"SKU1014","dc_id":"DC_SOF"}'
```

## If `/v2/readyz` fails

The message will now tell you exactly what is wrong, usually one of:

- `latest_features.csv` missing required feature columns
- `model.joblib` missing keys
- model `predict()` output does not contain P50/P90 values

In that case, rerun the fixed export notebook and download both generated files again.
