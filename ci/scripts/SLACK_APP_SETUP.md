# Publishing cuDNN Frontend Pipeline Monitor as a Slack App

This guide walks you through creating a proper Slack App for the pipeline monitor.

## Quick Setup (Incoming Webhook Only)

If you just want notifications without interactive features:

1. Go to https://api.slack.com/apps → **Create New App** → **From scratch**
2. Name: `cuDNN Frontend Pipeline Monitor`
3. Go to **Incoming Webhooks** → Enable → **Add New Webhook**
4. Select your channel → Copy the webhook URL
5. Use the webhook URL with `pipeline_slack_notifier.py`

---

## Full Slack App Setup

### Step 1: Create the App

1. Go to **https://api.slack.com/apps**
2. Click **"Create New App"** → **"From scratch"**
3. Configure:
   - **App Name:** `cuDNN Frontend Pipeline Monitor`
   - **Workspace:** Your NVIDIA Slack workspace
4. Click **Create App**

### Step 2: Configure Basic Info

Go to **Settings → Basic Information**:

1. Add a **Short Description:**
   ```
   Monitor cudnn_frontend GitLab pipeline status and get notified of new failures
   ```

2. Add an **App Icon** (optional):
   - Use a 512x512 PNG image
   - Suggested: cuDNN logo or a pipeline icon

3. Note your **Signing Secret** (under App Credentials)

### Step 3: Configure OAuth & Permissions

Go to **Features → OAuth & Permissions**:

1. Add **Bot Token Scopes:**
   - `chat:write` - Send messages as the bot
   - `chat:write.public` - Post to public channels without joining
   - `commands` - Add slash commands
   - `app_mentions:read` - React when mentioned (optional)

2. Click **Install to Workspace**
3. Authorize the app
4. Copy the **Bot User OAuth Token** (starts with `xoxb-`)

### Step 4: Add Incoming Webhooks

Go to **Features → Incoming Webhooks**:

1. Toggle **Activate Incoming Webhooks** to ON
2. Click **Add New Webhook to Workspace**
3. Select your channel (e.g., `#cudnn-frontend-ci`)
4. Copy the webhook URL

### Step 5: Add Slash Command (Optional)

Go to **Features → Slash Commands**:

1. Click **Create New Command**
2. Configure:
   ```
   Command: /pipeline-status
   Request URL: https://your-server.com/slack/commands
   Short Description: Check cudnn_frontend pipeline status
   Usage Hint: [branch]
   ```
3. Save

### Step 6: Enable Events API (Optional)

Go to **Features → Event Subscriptions**:

1. Toggle **Enable Events** to ON
2. Set **Request URL:** `https://your-server.com/slack/events`
3. Under **Subscribe to bot events**, add:
   - `app_mention` - React when @mentioned

### Step 7: Deploy the Server

For slash commands and events, you need a server:

```bash
# Install dependencies
pip install flask requests gunicorn

# Set environment variables
export SLACK_BOT_TOKEN="xoxb-your-bot-token"
export SLACK_SIGNING_SECRET="your-signing-secret"
export GITLAB_PRIVATE_TOKEN="glpat-your-token"

# Run locally for testing
python slack_app_server.py

# Run in production with gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 slack_app_server:app
```

### Deployment Options

#### Option A: Run on Internal Server

Deploy on an NVIDIA internal server:

```bash
# Using systemd service
sudo cat > /etc/systemd/system/pipeline-slack-bot.service << 'EOF'
[Unit]
Description=Pipeline Slack Bot
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/scripts
Environment=SLACK_BOT_TOKEN=xoxb-xxx
Environment=SLACK_SIGNING_SECRET=xxx
Environment=GITLAB_PRIVATE_TOKEN=glpat-xxx
ExecStart=/usr/bin/gunicorn -w 4 -b 0.0.0.0:8080 slack_app_server:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable pipeline-slack-bot
sudo systemctl start pipeline-slack-bot
```

#### Option B: Run on Cloud (AWS/GCP/Azure)

Use a serverless function or container:

**AWS Lambda + API Gateway:**
- Package the Flask app
- Deploy with Zappa or SAM

**Google Cloud Run:**
```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py .
CMD gunicorn -b :$PORT slack_app_server:app
EOF

# Deploy
gcloud run deploy pipeline-slack-bot \
  --source . \
  --set-env-vars SLACK_BOT_TOKEN=xxx,SLACK_SIGNING_SECRET=xxx,GITLAB_PRIVATE_TOKEN=xxx
```

#### Option C: Use ngrok for Testing

For local development:

```bash
# Terminal 1: Run the server
python slack_app_server.py

# Terminal 2: Expose with ngrok
ngrok http 8080

# Use the ngrok URL in Slack app settings
# e.g., https://abc123.ngrok.io/slack/commands
```

---

## Usage

### Using the Slash Command

In any Slack channel:

```
/pipeline-status           # Check develop branch
/pipeline-status main      # Check main branch
```

### Mentioning the Bot

If Events API is configured:

```
@Pipeline Monitor check status
@Pipeline Monitor status develop
```

### Automatic Notifications

The GitLab CI job will post automatically when pipelines complete.

---

## App Manifest (Alternative Setup)

Instead of manual configuration, you can use an app manifest:

```yaml
display_information:
  name: cuDNN Frontend Pipeline Monitor
  description: Monitor cudnn_frontend GitLab pipeline status
  background_color: "#76B900"
features:
  bot_user:
    display_name: Pipeline Monitor
    always_online: true
  slash_commands:
    - command: /pipeline-status
      url: https://your-server.com/slack/commands
      description: Check cudnn_frontend pipeline status
      usage_hint: "[branch]"
      should_escape: false
oauth_config:
  scopes:
    bot:
      - chat:write
      - chat:write.public
      - commands
      - app_mentions:read
settings:
  event_subscriptions:
    request_url: https://your-server.com/slack/events
    bot_events:
      - app_mention
  interactivity:
    is_enabled: false
  org_deploy_enabled: false
  socket_mode_enabled: false
  token_rotation_enabled: false
```

To use:
1. Go to https://api.slack.com/apps
2. Click **Create New App** → **From an app manifest**
3. Paste the YAML above
4. Update the URLs to your server

---

## Files

| File | Purpose |
|------|---------|
| `slack_app_server.py` | Flask server for slash commands & events |
| `pipeline_slack_notifier.py` | Webhook-based notifications (for CI) |
| `gitlab_pipeline_monitor.py` | Core monitoring logic |

---

## Troubleshooting

### "dispatch_failed" Error
- Your server didn't respond within 3 seconds
- Make sure to use background threads for long operations

### "invalid_auth" Error
- Check your SLACK_BOT_TOKEN is correct
- Make sure the app is installed to your workspace

### Messages Not Appearing
- Check the bot has been added to the channel
- Or use `chat:write.public` scope for public channels

### Slash Command Not Working
- Verify the Request URL is correct and accessible
- Check server logs for errors
- Ensure signing secret is correct

