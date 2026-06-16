# XPLORA Cloud Deployment Guide 🚀

This guide explains how to deploy the React frontend to **Vercel** and the FastAPI backend service to **Render** directly from your GitHub repository: `https://github.com/ryan1234814/XPLORA-Travel-Agent`.

---

## 📡 1. Deploy the Backend to Render

Render will read the `render.yaml` specification in the root of the repository to spin up a persistent Python Web Service.

### Steps:
1. Sign in to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** in the top right, then select **Blueprint**.
3. Connect your GitHub repository (`XPLORA-Travel-Agent`).
4. Render will auto-detect the service described in `render.yaml` (named `xplora-backend`).
5. Render will prompt you to input the values for the following synchronized environment variables:
   - `GROQ_API_KEY`: Your Groq platform key.
   - `OPENROUTER_API_KEY`: Your fallback OpenRouter key.
   - `TOMORROW_IO_API_KEY`: Your weather API key.
   - `SCRAPEGRAPH_API_KEY`: Your ScrapeGraphAI web scraper key.
   - `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`: Credentials for your remote MySQL Database (see Database Setup below).
6. Click **Apply**. Render will automatically provision the service, install requirements, and boot uvicorn.
7. **Copy the Web Service URL** (e.g. `https://xplora-backend.onrender.com`). You will need this for the Vercel frontend.

### 🗄️ Database Setup:
Since Render does not host MySQL natively for free, you can easily set up a free cloud MySQL instance:
- **Aiven.io** or **Clever Cloud** or **PlanetScale** host free MySQL DB instances.
- Simply provision a database instance, and copy the host, port, user, and password into the Render Blueprint environment variables.

---

## ⚡ 2. Deploy the Frontend to Vercel

Vercel will build and host your Vite/React static assets.

### Steps:
1. Sign in to your [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New** -> **Project**.
3. Import your GitHub repository (`XPLORA-Travel-Agent`).
4. In **Project Settings**:
   - Vercel automatically detects **Vite** as the framework.
   - **Root Directory**: Select `.` (Default).
   - **Build Command**: `npm run build` (Default).
   - **Output Directory**: `dist` (Default).
5. Open **Environment Variables** and add the following variable:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://your-backend-url.onrender.com` (paste your Render backend URL copied in Step 1).
6. Click **Deploy**. Vercel will install dependencies, build the client, and serve the application!

---

## 🔄 3. Continuous Integration
Whenever you push new changes to the `main` branch of your GitHub repository:
- **Vercel** will automatically rebuild and redeploy the frontend client.
- **Render** will automatically pull, install dependencies, and rebuild the FastAPI backend.
