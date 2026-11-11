# مِشْكَاة — Arabic-first frontend

## Start

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite development server proxies /api to http://127.0.0.1:8000.

## Production build

```bash
npm run build
```

**Current limitation:** The API intentionally returns HTTP 503 for diacritization until a trained model is installed. No mock results are presented as real inference.
