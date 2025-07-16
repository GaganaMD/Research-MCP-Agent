# scripts/deploy_aws.sh
#!/bin/bash

# Launch EC2 instance with GPU
aws ec2 run-instances \
    --image-id ami-0d70546e43a941d70 \
    --instance-type g4dn.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --user-data file://scripts/user_data.sh

# Install NVIDIA drivers and Docker
sudo apt-get update
sudo apt-get install -y nvidia-driver-470 nvidia-docker2
sudo systemctl restart docker

# Deploy application
docker run -d \
    --name mcp-research-agent \
    --gpus all \
    -p 8000:8000 \
    -v /data:/app/data \
    mcp-research-agent:latest