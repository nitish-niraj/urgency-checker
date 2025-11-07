# 🚀 Deployment Guide - Hugging Face Spaces

Complete step-by-step guide to deploy **Civic Issue Urgency Classifier** on Hugging Face Spaces using Docker.

---

## 📋 Prerequisites

1. ✅ **GitHub Account** - Repository must be on GitHub (public or private)
2. ✅ **Hugging Face Account** - Free account at https://huggingface.co
3. ✅ **Repository Ready** - All files pushed to GitHub (including Dockerfile)

---

## 🎯 Step-by-Step Deployment

### **Step 1: Prepare Your Repository**

Ensure all these files are in your GitHub repo:

```
urgency-checker/
├── Dockerfile                 ✅ Production container config
├── .dockerignore             ✅ Excludes unnecessary files
├── requirements.txt          ✅ HF-optimized dependencies
├── README_HF.md              ✅ HF-specific documentation
├── src/
│   ├── demo_api_browser.py   ✅ Main FastAPI app
│   └── ...
├── models/
│   ├── text_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── fusion_model.pkl
└── ...
```

**Command to check:**
```bash
git log --oneline | head -5  # Verify recent commits
git remote -v                # Verify origin is correct
```

### **Step 2: Create a Hugging Face Space**

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in the form:

   | Field | Value |
   |-------|-------|
   | **Space name** | `urgency-classifier` (or your choice) |
   | **License** | MIT (recommended) |
   | **Space SDK** | **Docker** ← Important! |
   | **Visibility** | Public (for demo) or Private |

4. Click **"Create Space"**

### **Step 3: Connect GitHub Repository**

After creating the Space:

1. Go to **"Space Settings"** (gear icon)
2. Scroll to **"Repository URL"**
3. Paste your GitHub repo URL:
   ```
   https://github.com/YOUR_USERNAME/urgency-checker.git
   ```
4. Under **"Docker"** section, select:
   - **Docker file location:** `Dockerfile` (default)
   - **Build context:** `/` (root)
5. Save settings

### **Step 4: Wait for Auto-Build**

HF Spaces will automatically:
1. Clone your GitHub repo
2. Build the Docker image from your Dockerfile
3. Deploy the container

**Monitor build progress:**
- Watch the "Build" tab in your Space
- Look for green checkmarks ✅
- First build takes 5-10 minutes

**If build fails:**
- Check the build logs
- Common issues: missing files, wrong paths, dependency conflicts
- Fix and push to GitHub - HF will rebuild automatically

### **Step 5: Test Your Deployment**

Once deployed (green status):

1. **Test the Web UI:**
   - Open your Space URL (e.g., `https://huggingface.co/spaces/YOUR_USERNAME/urgency-classifier`)
   - Enter a test civic issue
   - Click "Classify Urgency"
   - Verify results display correctly

2. **Test the API:**
   ```bash
   # Get your Space URL (e.g., https://huggingface.co/spaces/username/urgency-classifier)
   # Replace SPACE_URL with your actual Space URL
   
   curl -X POST https://SPACE_URL/classify-urgency \
     -H "Content-Type: application/json" \
     -d '{
       "text": "There are large, dangerous cracks in the road blocking ambulance access.",
       "location": "Hospital area",
       "category": "Infrastructure"
     }'
   ```

3. **Check API Docs:**
   - Navigate to: `https://SPACE_URL/docs`
   - Interactive Swagger UI with all endpoints
   - Try requests directly in the UI

### **Step 6: Verify Key Endpoints**

Test these endpoints to confirm everything works:

| Endpoint | Purpose | Expected |
|----------|---------|----------|
| `GET /` | Web UI | HTML page loads |
| `GET /health` | Health check | `{"status": "operational"}` |
| `GET /stats` | System stats | JSON with model info |
| `POST /classify-urgency` | Main classifier | Urgency level, score, confidence |

**Quick health check:**
```bash
curl https://SPACE_URL/health
```

Expected response:
```json
{
  "status": "operational",
  "models_loaded": true,
  "version": "1.0.0"
}
```

---

## 📊 Example API Calls (on Deployed Space)

Replace `SPACE_URL` with your actual Hugging Face Space URL.

### HIGH Priority Classification
```bash
curl -X POST https://SPACE_URL/classify-urgency \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Building facade crumbling - debris falling on pedestrians - URGENT",
    "location": "Downtown Square",
    "category": "Safety"
  }'
```

**Expected response:**
```json
{
  "urgency_level": "HIGH",
  "urgency_score": 9.2,
  "confidence": 0.97,
  "recommended_department": "Emergency Services",
  "estimated_response_time": "Within 30 minutes",
  "reasoning": "Critical safety issue - immediate action required"
}
```

### MEDIUM Priority Classification
```bash
curl -X POST https://SPACE_URL/classify-urgency \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Street light flickering at night; poses safety concern",
    "location": "Main Street",
    "category": "Infrastructure"
  }'
```

### LOW Priority Classification
```bash
curl -X POST https://SPACE_URL/classify-urgency \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Minor paint fading on park bench; cosmetic issue",
    "location": "Central Park",
    "category": "Maintenance"
  }'
```

---

## 🔄 Continuous Deployment

Once connected, every push to GitHub triggers automatic redeploy:

```bash
# Make changes locally
vim src/demo_api_browser.py

# Commit and push
git add .
git commit -m "Fix: improve error handling"
git push origin main

# HF Spaces automatically rebuilds within seconds!
```

**To disable auto-rebuild:** Go to Space Settings → uncheck "Build on push"

---

## ⚙️ Environment Variables (Optional)

To set environment variables in your Space:

1. Go to **Space Settings** (gear icon)
2. Scroll to **"Repository secrets"** (not visible until you add one)
3. Click **"New secret"**
4. Add variables like:
   - `LOG_LEVEL=DEBUG`
   - `MODEL_CACHE_TTL=3600`

The Dockerfile accesses them via `ENV` or `ARG`.

---

## 🐛 Troubleshooting

### Build Fails

**Error:** `Docker build failed`
- **Fix:** Check Dockerfile syntax, verify all files are in repo, check file paths

**Error:** `ModuleNotFoundError: No module named 'sklearn'`
- **Fix:** Ensure `scikit-learn==1.3.2` is in requirements.txt

**Error:** `Port 7860 already in use`
- **Fix:** Dockerfile specifies port 7860 - HF Spaces expects this

### Runtime Errors

**Error:** `Models not found`
- **Fix:** Ensure `/models/*.pkl` files are in Git LFS or repo

**Error:** `API endpoint returns 404`
- **Fix:** Check FastAPI app is listening on port 7860 in Dockerfile CMD

**Error:** `Timeout on classification`
- **Fix:** May indicate model loading issue or slow hardware (HF free tier has limits)

### Viewing Logs

1. Go to your Space
2. Click the **"Logs"** tab (bottom left)
3. Scroll through Docker build and runtime logs
4. Look for errors marked in red

---

## 📈 Monitoring & Analytics

After deployment:

1. **View Space Traffic:**
   - Go to Space Settings → "Space URL"
   - Check views and usage

2. **Monitor API Calls:**
   - Check `/stats` endpoint periodically
   - Look for anomalies in request counts

3. **Check Uptime:**
   - HF Spaces provides basic monitoring
   - Free tier may have auto-sleep on inactivity

---

## 🔐 Security Considerations

- ✅ **No sensitive data in Dockerfile** - Don't hardcode API keys
- ✅ **CORS enabled** - Frontend can call API
- ✅ **Input validation** - 10-5000 character limit on text
- ✅ **Rate limiting** - Consider adding if API gets abused

---

## 🎯 Next Steps After Deployment

1. **Share your Space URL** - Post on GitHub, social media, portfolio
2. **Get feedback** - Users can test and report issues
3. **Iterate** - Make improvements, push to GitHub, auto-redeploy
4. **Add features** - Image analysis, more categories, multi-language support

---

## 📚 Useful Links

- **Hugging Face Spaces Docs:** https://huggingface.co/docs/hub/spaces
- **Docker on HF Spaces:** https://huggingface.co/docs/hub/spaces-overview#docker
- **FastAPI Deployment:** https://fastapi.tiangolo.com/deployment/concepts/
- **Troubleshooting HF Spaces:** https://huggingface.co/docs/hub/spaces-troubleshooting

---

## ✅ Deployment Checklist

Before deploying, verify:

- [ ] All files committed to GitHub
- [ ] Dockerfile exists in repo root
- [ ] requirements.txt has all dependencies
- [ ] Models are in /models directory
- [ ] README_HF.md is complete
- [ ] No sensitive data in code
- [ ] .dockerignore excludes large unnecessary files
- [ ] API listens on port 7860
- [ ] CORS middleware enabled in FastAPI
- [ ] Health check endpoint works
- [ ] Classification endpoint has error handling

---

## 🚀 You're Ready!

Once all checks pass, create your HF Space and share the URL. Your AI classifier is live! 🎉

For questions or issues, refer to:
- README_HF.md - Usage documentation
- Main README.md - Technical details
- GitHub Issues - Report bugs

**Happy deploying!** 🌟
