#!/usr/bin/env python3
"""
Pipeline Monitoring Agent for AI Assistants

This is a simplified agent interface designed to be called by AI assistants
to check on cudnn_frontend nightly pipeline status.

Example usage from AI assistant:
    result = check_pipeline_status()
    if result["has_new_failures"]:
        print("Alert: New failures detected!")
        for failure in result["new_failures"]:
            print(f"  - {failure['name']}: {failure['url']}")
"""

import os
import sys
from typing import Dict, List, Optional, Any

# Add the scripts directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gitlab_pipeline_monitor import GitLabPipelineMonitor


def check_pipeline_status(
    token: Optional[str] = None, ref: str = "develop", verbose: bool = False
) -> Dict[str, Any]:
    """
    Check the status of the latest nightly pipeline runs.

    This function compares the last two scheduled (nightly) pipeline runs
    and identifies new failures, fixed issues, and persistent problems.

    Args:
        token: GitLab private token (or use GITLAB_PRIVATE_TOKEN env var)
        ref: Branch to check (default: "develop")
        verbose: Enable verbose logging

    Returns:
        Dictionary containing:
        - has_new_failures: bool - True if there are new failures
        - new_failures: List of new failures with name, stage, url
        - fixed_failures: List of failures that were fixed
        - persistent_failures: List of failures in both runs
        - current_pipeline: Info about the latest pipeline
        - previous_pipeline: Info about the previous pipeline
        - summary: Human-readable summary string

    Example:
        >>> result = check_pipeline_status()
        >>> print(result["summary"])
        >>> if result["has_new_failures"]:
        ...     for f in result["new_failures"]:
        ...         print(f"FAILED: {f['name']}")
    """
    try:
        monitor = GitLabPipelineMonitor(
            gitlab_url="https://gitlab-master.nvidia.com",
            project_path="cudnn/cudnn_frontend",
            private_token=token,
            verbose=verbose,
        )

        # Run the analysis (this prints detailed output)
        results = monitor.analyze_nightly_runs(ref=ref)

        # Add convenience fields
        results["has_new_failures"] = len(results.get("new_failures", [])) > 0
        results["has_persistent_failures"] = (
            len(results.get("persistent_failures", [])) > 0
        )

        # Create summary
        new_count = len(results.get("new_failures", []))
        fixed_count = len(results.get("fixed_failures", []))
        persistent_count = len(results.get("persistent_failures", []))

        if new_count > 0:
            summary = (
                f"⚠️ {new_count} NEW FAILURE(S) detected in the latest nightly run!"
            )
        elif persistent_count > 0:
            summary = (
                f"✅ No new failures. {persistent_count} persistent failure(s) remain."
            )
        else:
            summary = "✅ All tests passing! No failures in the latest nightly run."

        if fixed_count > 0:
            summary += f" ({fixed_count} failure(s) were fixed)"

        results["summary"] = summary

        return results

    except ValueError as e:
        return {"error": str(e), "has_new_failures": False, "summary": f"Error: {e}"}
    except Exception as e:
        return {
            "error": str(e),
            "has_new_failures": False,
            "summary": f"Error checking pipeline: {e}",
        }


def get_new_failures(token: Optional[str] = None, ref: str = "develop") -> List[Dict]:
    """
    Get only the new failures from the latest pipeline run.

    This is a convenience function that returns just the new failures.

    Args:
        token: GitLab private token
        ref: Branch to check

    Returns:
        List of new failures, each with name, stage, and url
    """
    result = check_pipeline_status(token=token, ref=ref, verbose=False)
    return result.get("new_failures", [])


def get_pipeline_summary(token: Optional[str] = None, ref: str = "develop") -> str:
    """
    Get a one-line summary of the pipeline status.

    Args:
        token: GitLab private token
        ref: Branch to check

    Returns:
        Human-readable summary string
    """
    result = check_pipeline_status(token=token, ref=ref, verbose=False)
    return result.get("summary", "Error getting pipeline status")


def format_failure_report(failures: List[Dict], title: str = "Failures") -> str:
    """
    Format a list of failures as a readable report.

    Args:
        failures: List of failure dictionaries
        title: Title for the report section

    Returns:
        Formatted string report
    """
    if not failures:
        return f"{title}: None\n"

    lines = [f"{title} ({len(failures)}):"]
    for f in sorted(failures, key=lambda x: (x.get("stage", ""), x.get("name", ""))):
        lines.append(f"  - [{f.get('stage', 'unknown')}] {f.get('name', 'unknown')}")
        if f.get("url"):
            lines.append(f"    URL: {f['url']}")

    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    print("Checking cudnn_frontend nightly pipeline status...")
    print("=" * 60)

    # Check if token is available
    if not os.environ.get("GITLAB_PRIVATE_TOKEN"):
        print("\nNote: Set GITLAB_PRIVATE_TOKEN environment variable")
        print("      or pass token parameter to functions")
        print("\nTo get a token:")
        print(
            "  1. Go to https://gitlab-master.nvidia.com/-/user_settings/personal_access_tokens"
        )
        print("  2. Create a token with 'read_api' scope")
        print("  3. export GITLAB_PRIVATE_TOKEN=your_token_here")
        sys.exit(1)

    result = check_pipeline_status()

    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(result["summary"])

    if result.get("has_new_failures"):
        print("\n" + format_failure_report(result["new_failures"], "🚨 NEW FAILURES"))

    if result.get("has_persistent_failures"):
        print(
            "\n"
            + format_failure_report(
                result["persistent_failures"], "⚠️ PERSISTENT FAILURES"
            )
        )
