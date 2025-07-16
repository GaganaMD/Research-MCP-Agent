# scripts/deploy_agent.py
import argparse
import subprocess
import yaml
from pathlib import Path
import logging

def main():
    parser = argparse.ArgumentParser(description="Deploy MCP Research Agent")
    parser.add_argument("--mode", choices=["local", "docker", "cloud"], default="local")
    parser.add_argument("--config", type=str, default="configs/agent_config.yaml")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    if args.mode == "local":
        deploy_local(args)
    elif args.mode == "docker":
        deploy_docker(args)
    elif args.mode == "cloud":
        deploy_cloud(args)

def deploy_local(args):
    """Deploy locally using uvicorn"""
    logging.info("Deploying locally...")
    
    # Start the server
    cmd = [
        "python", "-m", "uvicorn",
        "src.main:app",
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--reload"
    ]
    
    subprocess.run(cmd)

def deploy_docker(args):
    """Deploy using Docker"""
    logging.info("Building Docker image...")
    
    # Build image
    subprocess.run(["docker", "build", "-t", "mcp-research-agent", "-f", "docker/Dockerfile", "."])
    
    # Run container
    cmd = [
        "docker", "run", "-it",
        "--gpus", "all",
        "-p", f"{args.port}:8000",
        "-v", f"{Path.cwd()}/data:/app/data",
        "mcp-research-agent"
    ]
    
    subprocess.run(cmd)

def deploy_cloud(args):
    """Deploy to cloud (AWS/GCP/Azure)"""
    logging.info("Cloud deployment not yet implemented")
    logging.info("Please use Docker mode and deploy to your cloud provider")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()