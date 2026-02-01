# Clean up Docker system
docker system prune -f
docker builder prune -f

# Build with no cache to ensure fresh build
docker build --no-cache -t t3-challenge-dinov2 .