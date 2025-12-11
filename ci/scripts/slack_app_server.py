#!/usr/bin/env python3
"""
Slack App Server for Pipeline Monitor

This is a simple Flask server that handles Slack slash commands
and interactive features for the pipeline monitor.

Usage:
    # Install dependencies
    pip install flask requests gunicorn

    # Run locally for testing
    python slack_app_server.py

    # Run in production with gunicorn
    gunicorn -w 4 -b 0.0.0.0:8080 slack_app_server:app

Environment Variables Required:
    SLACK_BOT_TOKEN: Bot User OAuth Token (xoxb-...)
    SLACK_SIGNING_SECRET: App signing secret for verification
    GITLAB_PRIVATE_TOKEN: GitLab API token

Slack App Configuration:
    1. Create app at https://api.slack.com/apps
    2. Add Slash Command: /pipeline-status
    3. Set Request URL to: https://your-server.com/slack/commands
    4. Add Bot Token Scopes: chat:write, chat:write.public
    5. Install app to workspace
"""

import os
import sys
import json
import hmac
import hashlib
import time
from threading import Thread
from typing import Dict, Optional

try:
    from flask import Flask, request, jsonify
    import requests
except ImportError:
    print("Please install dependencies: pip install flask requests")
    sys.exit(1)

# Add script directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gitlab_pipeline_monitor import GitLabPipelineMonitor

app = Flask(__name__)

# Configuration
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
GITLAB_TOKEN = os.environ.get("GITLAB_PRIVATE_TOKEN")


def verify_slack_signature(request) -> bool:
    """Verify that the request came from Slack."""
    if not SLACK_SIGNING_SECRET:
        return True  # Skip verification if not configured

    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    # Check timestamp to prevent replay attacks
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False

    # Compute expected signature
    sig_basestring = f"v0:{timestamp}:{request.get_data(as_text=True)}"
    expected_sig = (
        "v0="
        + hmac.new(
            SLACK_SIGNING_SECRET.encode(), sig_basestring.encode(), hashlib.sha256
        ).hexdigest()
    )

    return hmac.compare_digest(expected_sig, signature)


def send_slack_message(channel: str, text: str, blocks: Optional[list] = None) -> bool:
    """Send a message to Slack using the Bot API."""
    if not SLACK_BOT_TOKEN:
        print("No SLACK_BOT_TOKEN configured")
        return False

    payload = {
        "channel": channel,
        "text": text,
    }
    if blocks:
        payload["blocks"] = blocks

    response = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            "Content-Type": "application/json",
        },
        json=payload,
    )

    result = response.json()
    if not result.get("ok"):
        print(f"Slack API error: {result.get('error')}")
        return False
    return True


def check_pipeline_and_respond(response_url: str, channel: str, ref: str = "develop"):
    """Check pipeline status and send response to Slack."""
    try:
        monitor = GitLabPipelineMonitor(
            gitlab_url="https://gitlab-master.nvidia.com",
            project_path="cudnn/cudnn_frontend",
            private_token=GITLAB_TOKEN,
            verbose=False,
        )

        # Get pipelines
        pipelines = monitor.get_scheduled_pipelines(ref=ref, count=2)

        if not pipelines:
            send_response(response_url, f"No scheduled pipelines found for `{ref}`")
            return

        current = pipelines[0]
        previous = pipelines[1] if len(pipelines) > 1 else None

        # Get jobs
        current.jobs = monitor.get_pipeline_jobs(current.id)
        if previous:
            previous.jobs = monitor.get_pipeline_jobs(previous.id)

        # Compare
        if previous:
            new_failures, fixed, persistent = monitor.compare_pipelines(
                current, previous
            )
        else:
            new_failures = current.failed_jobs
            fixed = []
            persistent = []

        # Build response
        lines = []
        lines.append("=" * 40)
        lines.append("*GitLab Pipeline Monitor - cudnn_frontend*")
        lines.append("=" * 40)
        lines.append(f"*Branch:* `{ref}`")
        lines.append(f"*Pipeline:* <{current.web_url}|#{current.id}>")
        lines.append(f"*Status:* `{current.status}`")
        lines.append("")

        # Results
        lines.append(f"🚨 *NEW FAILURES ({len(new_failures)}):*")
        if new_failures:
            for j in new_failures[:5]:
                lines.append(f"   ❌ `{j.name}`")
        else:
            lines.append("   ✅ No new failures!")

        lines.append("")
        lines.append(f"⚠️  *PERSISTENT ({len(persistent)}):*")
        if persistent:
            for j in persistent[:5]:
                lines.append(f"   🔴 `{j.name}`")
        else:
            lines.append("   (none)")

        lines.append("")
        lines.append(f"✅ *FIXED ({len(fixed)}):*")
        if fixed:
            for j in fixed[:3]:
                lines.append(f"   🔧 `{j.name}`")
        else:
            lines.append("   (none)")

        lines.append("")
        lines.append("=" * 40)
        lines.append(f"_{len(current.failed_jobs)} failed / {len(current.jobs)} total_")

        text = "\n".join(lines)

        # Determine color
        if new_failures:
            color = "danger"
        elif persistent:
            color = "warning"
        else:
            color = "good"

        send_response(response_url, text, color=color)

    except Exception as e:
        send_response(response_url, f"❌ Error checking pipeline: {str(e)}")


def send_response(response_url: str, text: str, color: str = None):
    """Send a response to Slack's response_url."""
    payload = {
        "response_type": "in_channel",  # or "ephemeral" for private
        "text": text,
    }

    if color:
        payload["attachments"] = [{"color": color, "text": text, "mrkdwn_in": ["text"]}]
        payload["text"] = ""  # Clear main text when using attachment

    requests.post(response_url, json=payload)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/slack/commands", methods=["POST"])
def handle_slash_command():
    """Handle Slack slash commands."""

    # Verify request is from Slack
    if not verify_slack_signature(request):
        return jsonify({"error": "Invalid signature"}), 401

    # Parse command
    command = request.form.get("command", "")
    text = request.form.get("text", "").strip()
    channel_id = request.form.get("channel_id", "")
    response_url = request.form.get("response_url", "")

    if command == "/pipeline-status":
        # Get branch from argument or default to develop
        ref = text if text else "develop"

        # Send immediate response
        immediate_response = {
            "response_type": "ephemeral",
            "text": f"🔍 Checking pipeline status for `{ref}`...",
        }

        # Process in background thread (Slack requires response within 3 seconds)
        thread = Thread(
            target=check_pipeline_and_respond, args=(response_url, channel_id, ref)
        )
        thread.start()

        return jsonify(immediate_response)

    return jsonify({"text": f"Unknown command: {command}"})


@app.route("/slack/events", methods=["POST"])
def handle_events():
    """Handle Slack Events API (for bot mentions, etc.)."""
    data = request.json

    # Handle URL verification challenge
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge")})

    # Handle other events
    event = data.get("event", {})
    event_type = event.get("type")

    if event_type == "app_mention":
        # Bot was mentioned
        channel = event.get("channel")
        text = event.get("text", "")

        if "status" in text.lower() or "check" in text.lower():
            # Check pipeline
            thread = Thread(target=lambda: check_and_post(channel, "develop"))
            thread.start()

    return jsonify({"ok": True})


def check_and_post(channel: str, ref: str):
    """Check pipeline and post to channel."""
    # Similar to check_pipeline_and_respond but uses chat.postMessage
    try:
        monitor = GitLabPipelineMonitor(
            gitlab_url="https://gitlab-master.nvidia.com",
            project_path="cudnn/cudnn_frontend",
            private_token=GITLAB_TOKEN,
            verbose=False,
        )

        pipelines = monitor.get_scheduled_pipelines(ref=ref, count=2)
        if not pipelines:
            send_slack_message(channel, f"No scheduled pipelines found for `{ref}`")
            return

        current = pipelines[0]
        current.jobs = monitor.get_pipeline_jobs(current.id)

        text = f"📊 *Pipeline #{current.id}* | `{ref}` | <{current.web_url}|View>\n"
        text += f"Status: `{current.status}` | {len(current.failed_jobs)} failed / {len(current.jobs)} total"

        send_slack_message(channel, text)

    except Exception as e:
        send_slack_message(channel, f"❌ Error: {str(e)}")


if __name__ == "__main__":
    print("Starting Slack App Server...")
    print("Endpoints:")
    print("  GET  /health         - Health check")
    print("  POST /slack/commands - Slash command handler")
    print("  POST /slack/events   - Events API handler")
    print("")
    print("Required environment variables:")
    print(f"  SLACK_BOT_TOKEN: {'✅ Set' if SLACK_BOT_TOKEN else '❌ Not set'}")
    print(
        f"  SLACK_SIGNING_SECRET: {'✅ Set' if SLACK_SIGNING_SECRET else '❌ Not set'}"
    )
    print(f"  GITLAB_PRIVATE_TOKEN: {'✅ Set' if GITLAB_TOKEN else '❌ Not set'}")
    print("")

    app.run(host="0.0.0.0", port=8080, debug=True)
