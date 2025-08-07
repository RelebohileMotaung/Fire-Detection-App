# Fire Detection App with Prometheus & Grafana

A real-time fire and smoke detection system powered by YOLOv8, with integrated Prometheus metrics and Grafana dashboards for monitoring and alerting.

## 🚀 Features

- **Real-time Detection**: Live fire and smoke detection using YOLOv8
- **AI Verification**: Gemini AI-powered false positive reduction
- **Prometheus Metrics**: Comprehensive monitoring with custom metrics
- **Grafana Dashboards**: Visual monitoring and alerting
- **Email Alerts**: Automated email notifications for fire detection
- **Video Recording**: Automatic incident recording
- **WebSocket Support**: Real-time frame streaming
- **RESTful API**: Complete API for configuration and monitoring

## 🛠️ Prerequisites

- Docker
- Docker Compose

## 🏃 Quick Start

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View running containers
docker ps
```

This will start:
- Fire Detection App on port 8000
- Prometheus on port 9090
- Grafana on port 3000

### Manual Docker Setup

If you prefer to run services individually:

#### 1. Start Prometheus
```bash
docker run -d -p 9090:9090 \
  -v "c:/Users/Dept Of Edu/Downloads/Final/Fire-Detection-App/prometheus.yml:/etc/prometheus/prometheus.yml" \
  prom/prometheus
```

#### 2. Start Grafana
```bash
docker run -d -p 3000:3000 grafana/grafana
```

## 🔗 Accessing Services

- **Fire Detection API**: http://localhost:8000
- **Prometheus UI**: http://localhost:9090
- **Grafana UI**: http://localhost:3000

## 📊 Monitoring & Metrics

### Prometheus Metrics Endpoint
The application exposes Prometheus metrics at:
```
http://localhost:8000/metrics
```

### Key Metrics Available
- `fire_detections_total`: Total number of fire detections
- `frame_processing_seconds`: Time taken to process each frame
- `email_alerts_sent_total`: Total email alerts sent
- `system_status`: Current system status
- `recording_status`: Recording status indicator

### Querying Metrics (PowerShell Examples)

```powershell
# Get fire detection metrics
curl http://localhost:8000/metrics -UseBasicParsing | Select-String "fire_detections_total"

# Get all metrics
curl http://localhost:8000/metrics -UseBasicParsing
```

### Grafana Configuration
1. Access Grafana at http://localhost:3000
2. Default credentials: admin/admin
3. Add Prometheus data source: http://prometheus:9090
4. Import dashboards or create custom visualizations

## 🛑 Stopping Services

### Docker Compose
```bash
docker-compose down
```

### Manual Docker
```bash
# Stop Prometheus
docker stop $(docker ps -q --filter ancestor=prom/prometheus)

# Stop Grafana
docker stop $(docker ps -q --filter ancestor=grafana/grafana)
```

## 📁 Project Structure

```
Fire-Detection-App/
├── backend_integrated_updated_complete.py  # Main backend application
├── frontend_integrated.py                  # Frontend interface
├── prometheus.yml                          # Prometheus configuration
├── docker-compose.yml                      # Docker services configuration
├── Dockerfile                              # App container configuration
├── requirements.txt                        # Python dependencies
├── best.pt                                 # YOLO model weights
├── recordings/                             # Incident recordings
└── README.md                              # This file
```

## 🔧 Configuration

### Environment Variables
- `EMAIL_USER`: SMTP email username
- `EMAIL_PASS`: SMTP email password
- `GEMINI_API_KEY`: Google Gemini API key
- `SMTP_SERVER`: SMTP server address
- `SMTP_PORT`: SMTP server port

### Prometheus Configuration
The `prometheus.yml` file is pre-configured to:
- Scrape metrics from the fire detection app
- Monitor Prometheus itself
- Set appropriate scrape intervals

## 🚨 Troubleshooting

### Common Issues
1. **Port conflicts**: Ensure ports 8000, 9090, and 3000 are available
2. **Docker permissions**: Run Docker Desktop as administrator on Windows
3. **Volume mounting**: Ensure the prometheus.yml path is correct in Docker commands

### Checking Service Status
```bash
# Check all running containers
docker ps

# Check container logs
docker logs <container_name>

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

## 📞 Support

For issues or questions, please check the logs or open an issue in the project repository.
