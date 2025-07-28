# KHÁI QUÁT PIPELINE 2 VERSION HỆ THỐNG CHẤM CÔNG AI

## 🎯 PIPELINE VERSION 1 (THEO TÀI LIỆU GỐC)

### **Kiến Trúc Tổng Quan V1**
```
Camera → Frigate → MQTT → Attendance Worker → AI Service → PostgreSQL/pgvector → Slack/Teams
```

### **Chi Tiết Pipeline V1**

#### **1. Video Processing Layer**
```
📹 IP Camera (RTSP)
    ↓ (dual-stream)
🎥 Frigate NVR (BẮT BUỘC)
    ├─ Sub-stream (480p) → Person Detection (CPU/Coral TPU)
    └─ Main-stream (1080p) → High-quality Snapshot
    ↓ (person detected in zone)
📤 MQTT Publish Event
```

#### **2. Event Processing Layer**
```
🔄 MQTT Broker (Eclipse Mosquitto)
    ↓ (subscribe events)
⚙️ Attendance Worker (Python)
    ├─ Download Snapshot từ Frigate
    ├─ Business Logic (cooldown, work hours)
    └─ Call AI Service
```

#### **3. AI Recognition Layer**
```
🤖 AI Service (Python + GPU)
    ├─ SCRFD Face Detection (BẮT BUỘC)
    ├─ ArcFace Embedding 512-dim (BẮT BUỘC)
    ├─ Basic Liveness Check
    └─ Return Results
```

#### **4. Data & Notification Layer**
```
🗄️ PostgreSQL + pgvector
    ├─ Vector Similarity Search (cosine distance)
    ├─ Attendance Record Creation
    └─ Business Logic Queries
    ↓
📱 Notification Services
    ├─ Slack API (Direct Messages)
    └─ Teams Webhook
```

### **Performance Characteristics V1**
- **End-to-end latency**: 3-8 seconds
- **Face detection**: SCRFD (~100-200ms)
- **Face recognition**: ArcFace (~50-100ms)
- **Database query**: pgvector (~10-50ms)
- **Scalability**: 1-5 cameras, 100-200 employees
- **Infrastructure**: Single server or simple cluster

---

## 🚀 PIPELINE VERSION 2 (CLOUD-NATIVE ENHANCED)

### **Kiến Trúc Tổng Quan V2**
```
Camera → Video Processor → NATS/Kafka → AI Microservices → Qdrant → Real-time Dashboard
                                      ↓
                               Kubernetes Cluster (Multi-GPU)
```

### **Chi Tiết Pipeline V2**

#### **1. Enhanced Video Processing**
```
📹 Multiple IP Cameras (4K/1080p)
    ↓ (WebRTC/RTSP)
🎬 Cloud-Native Video Processor (Rust/Go)
    ├─ Real-time Stream Processing
    ├─ Edge Computing Integration
    ├─ Multi-zone Detection
    └─ Load Balancing Across Cameras
    ↓ (intelligent buffering)
⚡ NATS JetStream (Sub-ms latency)
```

#### **2. AI Microservices Architecture**
```
🌐 Kubernetes Cluster với GPU Operator
    ├─ Face Detection Service (YOLO-FaceV2)
    │   ├─ 98.6% accuracy trên WiderFace
    │   ├─ Optimized for occlusion/small faces
    │   └─ 4-8 concurrent streams per GPU
    │
    ├─ Face Recognition Service (LVFace ViT)
    │   ├─ Vision Transformer architecture
    │   ├─ 94.31% accuracy trên challenging datasets
    │   └─ 512-dim embeddings
    │
    ├─ Advanced Anti-Spoofing Service
    │   ├─ Multi-modal detection (depth + texture + motion)
    │   ├─ Depth estimation với MiDaS
    │   ├─ Moiré pattern detection
    │   └─ Motion analysis across frames
    │
    └─ NVIDIA Triton Inference Server
        ├─ 40x performance vs TensorFlow Serving
        ├─ Dynamic batching
        ├─ TensorRT optimization
        └─ 272 inferences/second
```

#### **3. High-Performance Messaging**
```
📨 Hybrid Messaging Architecture
    ├─ NATS (Real-time events, <1ms latency)
    │   └─ 160K messages/second
    └─ Apache Kafka (High throughput data)
        └─ 1.9GB/second video processing
```

#### **4. Advanced Data Layer**
```
🔍 Vector Database (Qdrant - Rust-based)
    ├─ Sub-1ms query latency
    ├─ 4x RPS improvement vs alternatives
    ├─ Millions of face embeddings
    └─ Advanced filtering capabilities
    
📊 Caching Layer (Redis Cluster)
    ├─ Hot data caching
    ├─ Session management
    └─ Real-time counters
    
🗄️ PostgreSQL (Transactional data)
    ├─ Employee management
    ├─ Attendance logs
    └─ Audit trails
```

#### **5. Cloud-Native Features**
```
☸️ Kubernetes Advanced Features
    ├─ GPU Time-slicing (7 instances per GPU)
    ├─ Auto-scaling based on load
    ├─ Service Mesh (Linkerd - 8% latency overhead)
    └─ GitOps deployment (ArgoCD)
    
🛡️ Security & Compliance
    ├─ Zero-trust architecture
    ├─ Biometric data encryption
    ├─ FIDO2/WebAuthn authentication
    └─ GDPR/AI Act compliance
    
📊 Observability Stack
    ├─ OpenTelemetry + SigNoz
    ├─ Real-time drift detection
    ├─ Performance monitoring
    └─ Cost optimization tracking
```

#### **6. Modern Frontend**
```
💻 Next.js 14 + React 18
    ├─ Server Components + PPR
    ├─ Real-time WebSocket updates
    ├─ WebRTC live video streaming
    ├─ Progressive Web App capabilities
    └─ Biometric authentication (WebAuthn)
```

---

## 📊 SO SÁNH PERFORMANCE 2 VERSION

| **Metric** | **Version 1 (V1)** | **Version 2 (V2)** |
|------------|-------------------|-------------------|
| **End-to-End Latency** | 3-8 seconds | <1-3 seconds |
| **Face Detection** | SCRFD 100-200ms | YOLO-FaceV2 50-150ms |
| **Face Recognition** | ArcFace 50-100ms | LVFace <1ms |
| **Database Query** | pgvector 10-50ms | Qdrant <1ms |
| **Concurrent Cameras** | 1-5 cameras | 10-50+ cameras |
| **User Capacity** | 100-200 employees | 100-500+ employees |
| **Accuracy** | ~95% | ~98.6% |
| **Infrastructure** | Single server | Kubernetes cluster |
| **Scalability** | Vertical | Horizontal |
| **Cost (500 users)** | ~$5-8K/month | ~$8-15K/month |

---

## 🔄 MIGRATION PATH: V1 → V2

### **Phase 1: Infrastructure Modernization**
```
1. Setup Kubernetes cluster với GPU support
2. Deploy service mesh (Linkerd)
3. Migrate database to Qdrant + PostgreSQL cluster
4. Implement GitOps workflow
```

### **Phase 2: AI Model Upgrade**
```
1. Deploy YOLO-FaceV2 alongside SCRFD
2. A/B test performance và accuracy
3. Gradual migration to LVFace recognition
4. Enhanced anti-spoofing deployment
```

### **Phase 3: Microservices Migration**
```
1. Break monolithic AI service thành microservices
2. Implement NATS messaging
3. Deploy NVIDIA Triton serving
4. Real-time monitoring setup
```

### **Phase 4: Frontend Enhancement**
```
1. Upgrade to Next.js 14
2. Implement WebRTC streaming
3. Real-time dashboard với WebSockets
4. Progressive Web App features
```

### **Timeline Migration: 3-6 months**
- **Month 1-2**: Infrastructure setup
- **Month 3-4**: AI services migration
- **Month 5-6**: Frontend upgrade + optimization

---

## 🎯 KEY DIFFERENCES SUMMARY

| **Aspect** | **Version 1** | **Version 2** |
|------------|---------------|---------------|
| **Architecture** | Monolithic | Cloud-native microservices |
| **Deployment** | Docker Compose | Kubernetes |
| **AI Models** | SCRFD + ArcFace | YOLO-FaceV2 + LVFace ViT |
| **Database** | PostgreSQL only | Qdrant + PostgreSQL + Redis |
| **Messaging** | MQTT only | NATS + Kafka hybrid |
| **Frontend** | Basic React | Next.js 14 + WebRTC |
| **Security** | Basic authentication | Zero-trust + biometric encryption |
| **Monitoring** | Basic logging | Full observability stack |
| **Compliance** | GDPR basic | GDPR + AI Act ready |

**Version 1** phù hợp cho **proof-of-concept và small-medium deployments** với budget constraints.

**Version 2** phù hợp cho **enterprise production environments** yêu cầu high performance, scalability, và comprehensive compliance.