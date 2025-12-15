"""
CLI Tools for MLOps Management
Using Typer for command-line interface
"""

import typer
from typing import Optional
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.performance import performance_monitor
from monitoring.drift_detection import drift_detector
from versioning.canary import canary_deployment
from logging.config import mlops_logger

app = typer.Typer(help="MLOps CLI for Churn Prediction Model")

# Model Management Commands
model_app = typer.Typer(help="Model version management")
app.add_typer(model_app, name="model")

@model_app.command("list")
def list_models():
    """List all registered model versions"""
    status = canary_deployment.get_status()
    
    typer.echo("\n📦 Registered Models:")
    typer.echo("=" * 60)
    
    for version in status['registered_models']:
        is_primary = version == status['primary_version']
        is_canary = version == status['canary_version']
        
        badge = ""
        if is_primary:
            badge = typer.style(" [PRIMARY]", fg=typer.colors.GREEN, bold=True)
        elif is_canary:
            badge = typer.style(" [CANARY]", fg=typer.colors.YELLOW, bold=True)
        
        typer.echo(f"  • {version}{badge}")
        typer.echo(f"    Predictions: {status['prediction_counts'].get(version, 0)}")
    
    typer.echo()

@model_app.command("status")
def model_status():
    """Show current deployment status"""
    status = canary_deployment.get_status()
    
    typer.echo("\n🚀 Deployment Status:")
    typer.echo("=" * 60)
    typer.echo(f"Primary Version:  {status['primary_version']} ({status['traffic_split']['primary']}% traffic)")
    
    if status['canary_version']:
        typer.echo(f"Canary Version:   {status['canary_version']} ({status['traffic_split']['canary']}% traffic)")
    else:
        typer.echo("Canary Version:   None")
    
    typer.echo()

@model_app.command("deploy")
def deploy_canary(
    version: str = typer.Argument(..., help="Model version to deploy as canary"),
    traffic: int = typer.Option(10, help="Traffic percentage for canary (0-100)")
):
    """Deploy a model version as canary"""
    try:
        canary_deployment.set_canary(version, traffic)
        typer.echo(typer.style(f"✓ Deployed {version} as canary with {traffic}% traffic", fg=typer.colors.GREEN))
    except Exception as e:
        typer.echo(typer.style(f"✗ Error: {e}", fg=typer.colors.RED))
        raise typer.Exit(1)

@model_app.command("promote")
def promote_canary():
    """Promote canary to primary"""
    try:
        canary_deployment.promote_canary()
        typer.echo(typer.style("✓ Canary promoted to primary", fg=typer.colors.GREEN))
    except Exception as e:
        typer.echo(typer.style(f"✗ Error: {e}", fg=typer.colors.RED))
        raise typer.Exit(1)

@model_app.command("rollback")
def rollback_canary():
    """Rollback canary deployment"""
    try:
        canary_deployment.rollback_canary()
        typer.echo(typer.style("✓ Canary rolled back", fg=typer.colors.GREEN))
    except Exception as e:
        typer.echo(typer.style(f"✗ Error: {e}", fg=typer.colors.RED))
        raise typer.Exit(1)


# Monitoring Commands
monitor_app = typer.Typer(help="Performance and drift monitoring")
app.add_typer(monitor_app, name="monitor")

@monitor_app.command("metrics")
def show_metrics():
    """Show current performance metrics"""
    metrics = performance_monitor.get_metrics()
    
    typer.echo("\n📊 Performance Metrics:")
    typer.echo("=" * 60)
    typer.echo(f"Total Predictions:    {metrics['total_predictions']}")
    typer.echo(f"Accuracy:             {metrics['accuracy']:.2%}")
    typer.echo(f"Average Latency:      {metrics['average_latency']:.2f} ms")
    typer.echo(f"Min Latency:          {metrics['min_latency']:.2f} ms")
    typer.echo(f"Max Latency:          {metrics['max_latency']:.2f} ms")
    
    if metrics['time_range']['start'] and metrics['time_range']['end']:
        typer.echo(f"Time Range:           {metrics['time_range']['start']} to {metrics['time_range']['end']}")
    
    typer.echo()

@monitor_app.command("drift")
def check_drift():
    """Check for data drift"""
    drift_summary = drift_detector.get_drift_summary()
    
    typer.echo("\n🔍 Data Drift Analysis:")
    typer.echo("=" * 60)
    typer.echo(f"Total Features Tracked:   {drift_summary['total_features']}")
    typer.echo(f"Features with Drift:      {drift_summary['drifted_features_count']}")
    typer.echo(f"Drift Percentage:         {drift_summary['drift_percentage']:.1f}%")
    typer.echo(f"Current Samples:          {drift_summary['current_samples']}")
    
    if drift_summary['drifted_features']:
        typer.echo("\nDrifted Features:")
        for feature in drift_summary['drifted_features']:
            typer.echo(typer.style(f"  ⚠ {feature}", fg=typer.colors.YELLOW))
    else:
        typer.echo(typer.style("\n✓ No drift detected", fg=typer.colors.GREEN))
    
    typer.echo()

@monitor_app.command("reset")
def reset_monitoring():
    """Reset all monitoring data"""
    confirm = typer.confirm("Are you sure you want to reset all monitoring data?")
    
    if confirm:
        performance_monitor.reset()
        drift_detector.current_samples = []
        typer.echo(typer.style("✓ Monitoring data reset", fg=typer.colors.GREEN))
    else:
        typer.echo("Cancelled")


# Logs Commands
logs_app = typer.Typer(help="View and export logs")
app.add_typer(logs_app, name="logs")

@logs_app.command("view")
def view_logs(
    log_type: str = typer.Argument(..., help="Log type: prediction, performance, drift, error, system"),
    lines: int = typer.Option(20, help="Number of lines to show")
):
    """View recent logs"""
    log_files = {
        'prediction': 'predictions.log',
        'performance': 'performance.log',
        'drift': 'drift.log',
        'error': 'errors.log',
        'system': 'system.log'
    }
    
    if log_type not in log_files:
        typer.echo(typer.style(f"✗ Invalid log type. Choose from: {', '.join(log_files.keys())}", fg=typer.colors.RED))
        raise typer.Exit(1)
    
    log_path = Path(__file__).parent.parent / "logs" / log_files[log_type]
    
    if not log_path.exists():
        typer.echo(typer.style(f"✗ Log file not found: {log_path}", fg=typer.colors.RED))
        raise typer.Exit(1)
    
    # Read last N lines
    with open(log_path, 'r') as f:
        all_lines = f.readlines()
        recent_lines = all_lines[-lines:]
    
    typer.echo(f"\n📄 Recent {log_type.capitalize()} Logs ({len(recent_lines)} lines):")
    typer.echo("=" * 80)
    
    for line in recent_lines:
        typer.echo(line.rstrip())
    
    typer.echo()

@logs_app.command("export")
def export_logs(
    output_dir: str = typer.Option("./logs_export", help="Output directory for exported logs")
):
    """Export all logs to directory"""
    import shutil
    
    source_dir = Path(__file__).parent.parent / "logs"
    dest_dir = Path(output_dir)
    
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        for log_file in source_dir.glob("*.log*"):
            shutil.copy2(log_file, dest_dir / log_file.name)
        
        typer.echo(typer.style(f"✓ Logs exported to {dest_dir}", fg=typer.colors.GREEN))
    except Exception as e:
        typer.echo(typer.style(f"✗ Error: {e}", fg=typer.colors.RED))
        raise typer.Exit(1)


# Main info command
@app.command()
def info():
    """Show overall system information"""
    typer.echo("\n🎯 MLOps System Information")
    typer.echo("=" * 60)
    
    # Model status
    status = canary_deployment.get_status()
    typer.echo(f"\n📦 Models: {len(status['registered_models'])} registered")
    typer.echo(f"🚀 Primary: {status['primary_version']}")
    
    if status['canary_version']:
        typer.echo(f"🐤 Canary: {status['canary_version']} ({status['traffic_split']['canary']}% traffic)")
    
    # Performance
    metrics = performance_monitor.get_metrics()
    typer.echo(f"\n📊 Performance:")
    typer.echo(f"   Predictions: {metrics['total_predictions']}")
    typer.echo(f"   Accuracy: {metrics['accuracy']:.2%}")
    
    # Drift
    drift_summary = drift_detector.get_drift_summary()
    typer.echo(f"\n🔍 Drift Detection:")
    typer.echo(f"   Tracked Features: {drift_summary['total_features']}")
    typer.echo(f"   Drifted Features: {drift_summary['drifted_features_count']}")
    
    typer.echo()


if __name__ == "__main__":
    app()
