# 部署

```bash
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
streamlit run sm_app.py
```

## 如果采用PM2部署

```bash
pm2 start  streamlit run sm_app.py

```