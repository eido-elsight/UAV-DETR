#!/bin/bash
# Docker Training Script for UAV-DETR Halo Object Detection
# Simplified interface to run training in Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 UAV-DETR Docker Training for Halo Object Detection${NC}"
echo -e "${BLUE}=================================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  NVIDIA Docker support not detected. Training will use CPU only.${NC}"
    echo -e "${YELLOW}   For GPU training, install nvidia-docker2 package.${NC}"
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p runs weights evaluation_results

# Build Docker image
echo -e "${BLUE}🔨 Building Docker image...${NC}"
docker compose build

# Function to run training
run_training() {
    local model_variant=$1
    local epochs=${2:-100}
    
    echo -e "${GREEN}🎯 Starting UAV-DETR training with ${model_variant} variant${NC}"
    echo -e "${GREEN}   Epochs: ${epochs}${NC}"
    echo -e "${GREEN}   Output: ./runs/detect/train_${model_variant}${NC}"
    
    docker compose run --rm uav-detr-training python train_halo.py \
        --model ${model_variant} \
        --epochs ${epochs} \
        --name train_${model_variant}
}

# Function to run evaluation
run_evaluation() {
    local model_path=$1
    
    echo -e "${GREEN}📊 Running evaluation on model: ${model_path}${NC}"
    
    docker compose run --rm uav-detr-training python evaluate_halo.py \
        --model ${model_path} \
        --output evaluation_results/eval_$(basename ${model_path%.*})
}

# Function to run shell
run_shell() {
    echo -e "${GREEN}🐚 Starting interactive shell in container...${NC}"
    docker compose run --rm uav-detr-training /bin/bash
}

# Parse command line arguments
case "${1:-help}" in
    "train")
        model_variant=${2:-r18}
        epochs=${3:-100}
        
        if [[ ! "$model_variant" =~ ^(r18|r50)$ ]]; then
            echo -e "${RED}❌ Invalid model variant. Use 'r18' or 'r50'${NC}"
            exit 1
        fi
        
        run_training $model_variant $epochs
        ;;
    
    "eval")
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please provide model path. Example: ./docker_train.sh eval ./weights/best.pt${NC}"
            exit 1
        fi
        run_evaluation $2
        ;;
    
    "shell")
        run_shell
        ;;
    
    "clean")
        echo -e "${YELLOW}🧹 Cleaning Docker containers and images...${NC}"
        docker compose down
        docker image prune -f
        echo -e "${GREEN}✅ Cleanup complete${NC}"
        ;;
    
    "help"|*)
        echo -e "${BLUE}Usage: $0 <command> [options]${NC}"
        echo ""
        echo -e "${YELLOW}Commands:${NC}"
        echo -e "  ${GREEN}train <r18|r50> [epochs]${NC}  - Train UAV-DETR model"
        echo -e "                                  Default: r18, 100 epochs"
        echo -e "  ${GREEN}eval <model_path>${NC}           - Evaluate trained model"
        echo -e "  ${GREEN}shell${NC}                       - Start interactive shell"
        echo -e "  ${GREEN}clean${NC}                       - Clean Docker containers/images"
        echo -e "  ${GREEN}help${NC}                        - Show this help"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo -e "  $0 train r18 50          # Train R18 model for 50 epochs"
        echo -e "  $0 train r50             # Train R50 model for 100 epochs"
        echo -e "  $0 eval ./weights/best.pt # Evaluate trained model"
        echo -e "  $0 shell                 # Interactive development"
        echo ""
        echo -e "${YELLOW}Outputs:${NC}"
        echo -e "  Training: ./runs/detect/train_<variant>/"
        echo -e "  Weights: ./weights/"
        echo -e "  Evaluation: ./evaluation_results/"
        ;;
esac
