#!/bin/bash
# 📞 Telecom Customer Churn Prediction System - EC2 Deployment Script
# Targets: Ubuntu 22.04 LTS

set -e

echo "=========================================================="
echo "🚀 Starting Telecom Churn Prediction App Deployment..."
echo "=========================================================="

# 1. Update and Upgrade System Packages
echo "🔄 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "🐳 Docker not found. Installing Docker..."
    sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
    
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Set up the repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
      
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Configure current user to run Docker commands without sudo
    sudo usermod -aG docker $USER
    echo "✅ Docker installed successfully!"
else
    echo "✅ Docker is already installed."
fi

# 3. Build Docker Image
echo "🔨 Building Docker image (telecom-churn-app:latest)..."
sudo docker build -t telecom-churn-app:latest .

# 4. Stop and remove existing container if running
CONTAINER_NAME="telecom-churn-container"
if sudo docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}$"; then
    echo "🛑 Stopping and removing existing container..."
    sudo docker stop $CONTAINER_NAME || true
    sudo docker rm $CONTAINER_NAME || true
fi

# 5. Run the Container
echo "🏃 Running Docker container on port 8501..."
sudo docker run -d \
  --name $CONTAINER_NAME \
  -p 8501:8501 \
  --restart always \
  telecom-churn-app:latest

# 6. Output Status and Success
echo "=========================================================="
echo "🎉 Deployment Completed Successfully!"
echo "📈 Streamlit App is running in the background."
echo "🔗 Access it at: http://<your-ec2-public-ip>:8501"
echo "=========================================================="
