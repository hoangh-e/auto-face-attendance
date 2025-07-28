graph TB
    %% PIPELINE 1 (BAN ĐẦU) - THEO TÀI LIỆU GỐC
    subgraph "🎯 PIPELINE 1: VERSION THEO TÀI LIỆU GỐC"
        direction TB
        
        %% Employee Registration Flow
        subgraph REG1["👤 EMPLOYEE REGISTRATION PROCESS"]
            direction LR
            A1[👤 Admin Input<br/>Employee Info] --> B1[📸 Upload Face Images<br/>3-5 photos]
            B1 --> C1[🤖 SCRFD Detection<br/>Face detection]
            C1 --> D1[🧠 ArcFace Recognition<br/>512-dim embeddings]
            D1 --> E1[🗄️ PostgreSQL + pgvector<br/>Store employee + embeddings]
        end
        
        %% Main Pipeline Flow
        subgraph MAIN1["🎬 MAIN ATTENDANCE PIPELINE"]
            direction TB
            
            %% Video Input Layer
            F1[📹 IP Camera<br/>RTSP Stream] --> G1[🎥 Frigate NVR<br/>BẮT BUỘC]
            
            %% Frigate Processing
            subgraph FRIGATE1["Frigate Processing"]
                G1 --> H1[📺 Sub-stream 480p<br/>Person Detection]
                G1 --> I1[📺 Main-stream 1080p<br/>High Quality Snapshot]
                H1 --> J1{Person in Zone?}
                J1 -->|Yes| I1
                J1 -->|No| K1[⏭️ Skip Frame]
            end
            
            %% Event Processing
            I1 --> L1[📤 MQTT Publish<br/>Eclipse Mosquitto]
            L1 --> M1[⚙️ Attendance Worker<br/>Python Process]
            
            %% AI Processing
            M1 --> N1[📥 Download Snapshot<br/>từ Frigate]
            N1 --> O1[🤖 SCRFD Detection<br/>Face detection]
            O1 --> P1[🧠 ArcFace Recognition<br/>512-dim embedding]
            P1 --> Q1[🔍 Vector Search<br/>PostgreSQL pgvector]
            
            %% Business Logic
            Q1 --> R1{Employee Found?<br/>Similarity > 0.65}
            R1 -->|Yes| S1[⏰ Business Logic<br/>Cooldown + Work Hours]
            R1 -->|No| T1[❓ Unknown Person<br/>Log + Alert]
            
            S1 --> U1{Should Record?}
            U1 -->|Yes| V1[📝 Record Attendance<br/>PostgreSQL]
            U1 -->|No| W1[⏭️ Skip Recording<br/>Too Soon/Outside Hours]
            
            %% Notifications
            V1 --> X1[📱 Slack Notification<br/>Direct Message]
            V1 --> Y1[💬 Teams Notification<br/>Webhook]
            
            %% Dashboard
            V1 --> Z1[📊 Real-time Dashboard<br/>WebSocket Updates]
        end
        
        %% Performance Characteristics
        subgraph PERF1["⚡ PERFORMANCE V1"]
            direction LR
            P1A[⏱️ End-to-End: 3-8s<br/>🔧 Single Server<br/>📈 100-200 employees<br/>📷 1-5 cameras]
        end
    end

    %% PIPELINE 2 (CẢI TIẾN) - CLOUD-NATIVE
    subgraph "🚀 PIPELINE 2: CLOUD-NATIVE ENHANCED"
        direction TB
        
        %% Employee Registration Flow V2
        subgraph REG2["👤 EMPLOYEE REGISTRATION PROCESS V2"]
            direction LR
            A2[👤 Admin Panel<br/>Next.js 14 Interface] --> B2[📸 Upload Face Images<br/>WebRTC + Multiple formats]
            B2 --> C2[🤖 YOLO-FaceV2<br/>Enhanced detection]
            C2 --> D2[🧠 LVFace ViT<br/>Vision Transformer]
            D2 --> E2[🗄️ Qdrant Vector DB<br/>Rust-based, <1ms query]
        end
        
        %% Main Pipeline Flow V2
        subgraph MAIN2["🎬 CLOUD-NATIVE MICROSERVICES PIPELINE"]
            direction TB
            
            %% Video Input Layer V2
            F2[📹 Multiple IP Cameras<br/>4K Support] --> G2[🎬 Cloud Video Processor<br/>Rust/Go Microservice]
            
            %% Kubernetes AI Services
            subgraph K8S["☸️ Kubernetes GPU Cluster"]
                G2 --> H2[⚡ NATS JetStream<br/>Sub-ms latency]
                H2 --> I2[🤖 Face Detection Service<br/>YOLO-FaceV2]
                I2 --> J2[🧠 Face Recognition Service<br/>LVFace ViT]
                J2 --> K2[🛡️ Anti-Spoofing Service<br/>Multi-modal detection]
                K2 --> L2[🚀 NVIDIA Triton Server<br/>40x faster inference]
            end
            
            %% Enhanced Messaging
            L2 --> M2[📨 Hybrid Messaging<br/>NATS + Kafka]
            M2 --> N2[⚙️ Business Logic Service<br/>Advanced rules engine]
            
            %% Enhanced Data Layer
            N2 --> O2[🔍 Qdrant Vector Search<br/>Sub-1ms query]
            O2 --> P2{Employee Found?<br/>Advanced ML scoring}
            P2 -->|Yes| Q2[⏰ Smart Business Logic<br/>ML-based patterns]
            P2 -->|No| R2[🔔 Intelligent Alerts<br/>Context-aware]
            
            Q2 --> S2{Multi-factor Validation}
            S2 -->|Passed| T2[📝 Multi-DB Recording<br/>Qdrant + PostgreSQL + Redis]
            S2 -->|Failed| U2[🔄 Retry Logic<br/>Adaptive thresholds]
            
            %% Enhanced Notifications
            T2 --> V2[📱 Multi-Channel Notify<br/>Slack + Teams + Mobile]
            T2 --> W2[📊 Real-time Analytics<br/>Advanced dashboards]
            
            %% Service Mesh
            subgraph MESH["🕸️ Service Mesh (Linkerd)"]
                X2[🔐 Zero-Trust Security<br/>mTLS encryption]
                Y2[📊 Observability<br/>OpenTelemetry + SigNoz]
                Z2[⚖️ Load Balancing<br/>Intelligent routing]
            end
        end
        
        %% Performance Characteristics V2
        subgraph PERF2["⚡ PERFORMANCE V2"]
            direction LR
            P2A[⏱️ End-to-End: <1-3s<br/>☸️ Kubernetes Cluster<br/>📈 100-500+ employees<br/>📷 10-50+ cameras<br/>🔄 Auto-scaling]
        end
    end

    %% Comparison Arrows
    PERF1 -.->|UPGRADE| PERF2
    REG1 -.->|ENHANCE| REG2
    MAIN1 -.->|MODERNIZE| MAIN2

    %% Styling
    classDef pipeline1 fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef pipeline2 fill:#e8f5e8,stroke:#1b5e20,stroke-width:3px,color:#000
    classDef process1 fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef process2 fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef storage1 fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000
    classDef storage2 fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#000
    classDef ai1 fill:#fff8e1,stroke:#ff8f00,stroke-width:2px,color:#000
    classDef ai2 fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#000
    
    class REG1,MAIN1 pipeline1
    class REG2,MAIN2,K8S,MESH pipeline2
    class A1,B1,M1,N1,S1,U1 process1
    class A2,B2,N2,Q2,S2 process2
    class E1,L1,Q1,V1 storage1
    class E2,M2,O2,T2 storage2
    class C1,D1,O1,P1 ai1
    class C2,D2,I2,J2,K2,L2 ai2