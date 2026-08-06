# Diagnostic Gradio client

This UI lets a human answer the local environment's binary-choice episodes and
inspect the transparent reward components. It is not a trained-model demo and
does not display benchmark claims.

Start the API first, then run:

```powershell
$env:SYCO_SERVER_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe deploy\demo-space\app.py
```

For a Hugging Face Space, the root `Dockerfile` packages this app and
`SYCO_SERVER_URL` must point to a separately deployed API. Do not expose an
unauthenticated private API merely to make the demo public.
