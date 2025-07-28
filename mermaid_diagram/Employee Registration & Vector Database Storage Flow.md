graph TD
    %% EMPLOYEE REGISTRATION & VECTOR DATABASE FLOW
    subgraph "👤 EMPLOYEE REGISTRATION & VECTOR DATABASE PROCESS"
        direction TB
        
        %% Input Phase
        subgraph INPUT["📝 INPUT PHASE"]
            A[👨‍💼 Admin Access<br/>Web Admin Panel] --> B[📋 Employee Form<br/>Name, Email, Dept, Position]
            B --> C[📸 Face Image Upload<br/>3-5 clear photos<br/>Different angles]
        end
        
        %% Image Processing Phase
        subgraph PROCESS["🔧 IMAGE PROCESSING PHASE"]
            C --> D[📊 Image Validation<br/>Format, Size, Quality Check]
            D --> E{Valid Images?}
            E -->|No| F[❌ Error Message<br/>Upload Again]
            E -->|Yes| G[🔄 Pre-processing<br/>Resize, Normalize, Enhance]
            F --> C
        end
        
        %% AI Processing Phase - Pipeline 1
        subgraph AI1["🤖 AI PROCESSING - PIPELINE 1"]
            G --> H1[🎯 SCRFD Detection<br/>Detect faces in each image]
            H1 --> I1{Faces Detected?}
            I1 -->|No Face| J1[⚠️ Warning: No face found<br/>Skip this image]
            I1 -->|Face Found| K1[🧠 ArcFace Recognition<br/>Extract 512-dim embedding]
            K1 --> L1[📏 Quality Assessment<br/>Sharpness, Brightness, Size]
            L1 --> M1{Quality Score > 0.3?}
            M1 -->|No| N1[⚠️ Low quality image<br/>Skip or retry]
            M1 -->|Yes| O1[✅ Valid Embedding<br/>Store for processing]
        end
        
        %% AI Processing Phase - Pipeline 2  
        subgraph AI2["🚀 AI PROCESSING - PIPELINE 2"]
            G --> H2[🎯 YOLO-FaceV2<br/>Enhanced face detection]
            H2 --> I2{Faces Detected?}
            I2 -->|No Face| J2[⚠️ AI-powered retry<br/>Different preprocessing]
            I2 -->|Face Found| K2[🧠 LVFace ViT<br/>Vision Transformer embedding]
            K2 --> L2[🛡️ Advanced Quality Check<br/>Liveness, Anti-spoofing]
            L2 --> M2{Multi-factor Validation?}
            M2 -->|Failed| N2[🔄 Smart retry<br/>Adjust parameters]
            M2 -->|Passed| O2[✅ High-quality Embedding<br/>Enhanced features]
        end
        
        %% Embedding Processing
        subgraph EMBEDDING["🧮 EMBEDDING PROCESSING"]
            O1 --> P[📊 Collect All Embeddings<br/>From valid images]
            O2 --> P
            P --> Q[🔢 Multiple Embedding Strategy]
            
            subgraph STRATEGY["Processing Strategy"]
                Q --> R1[📈 Average Embedding<br/>Simple mean calculation]
                Q --> R2[⚖️ Weighted Average<br/>Quality-based weights]
                Q --> R3[🎯 Best Embedding<br/>Highest quality score]
                Q --> R4[📚 Store All Embeddings<br/>Individual + combined]
            end
            
            R1 --> S[🧪 Final Embedding<br/>512-dimensional vector]
            R2 --> S
            R3 --> S
            R4 --> S
        end
        
        %% Database Storage - Pipeline 1
        subgraph DB1["🗄️ DATABASE STORAGE - PIPELINE 1"]
            S --> T1[📝 Employee Record<br/>PostgreSQL main table]
            T1 --> U1[🔢 Face Embedding<br/>pgvector column]
            U1 --> V1[📸 Face Registrations<br/>Individual image records]
            V1 --> W1[🔗 Create Indexes<br/>Vector similarity indexes]
            W1 --> X1[✅ Registration Complete<br/>Employee ready for recognition]
        end
        
        %% Database Storage - Pipeline 2
        subgraph DB2["🚀 DATABASE STORAGE - PIPELINE 2"]
            S --> T2[📝 Multi-DB Strategy<br/>PostgreSQL + Qdrant]
            T2 --> U2[⚡ Qdrant Vector Store<br/>High-performance search]
            U2 --> V2[🗄️ PostgreSQL Relations<br/>Employee metadata]
            V2 --> W2[💾 Redis Cache<br/>Hot embeddings]
            W2 --> X2[🔧 Auto-optimization<br/>Vector index tuning]
            X2 --> Y2[✅ Enhanced Registration<br/>Sub-1ms search ready]
        end
        
        %% Validation & Testing
        subgraph VALIDATION["🧪 VALIDATION & TESTING"]
            X1 --> Z[🎯 Test Recognition<br/>Upload test image]
            Y2 --> Z
            Z --> AA[🔍 Similarity Search<br/>Find closest match]
            AA --> BB{Recognition Success?}
            BB -->|Yes| CC[✅ Registration Confirmed<br/>Threshold validation passed]
            BB -->|No| DD[⚠️ Adjust Threshold<br/>Or re-register with better images]
            CC --> EE[📊 Generate Report<br/>Registration summary]
            DD --> C
        end
        
        %% Performance Metrics
        subgraph METRICS["📊 PERFORMANCE METRICS"]
            EE --> FF[⏱️ Processing Time<br/>Total registration duration]
            FF --> GG[🎯 Quality Scores<br/>Average image quality]
            GG --> HH[🔢 Embedding Stats<br/>Vector similarity distribution]
            HH --> II[📈 Success Rate<br/>Recognition accuracy]
        end
    end
    
    %% Error Handling Flow
    subgraph ERROR["🚨 ERROR HANDLING"]
        JJ[❌ Registration Failed] --> KK{Error Type}
        KK -->|Image Quality| LL[📸 Request better images<br/>Lighting, angle guidance]
        KK -->|No Face Detected| MM[🎯 Face positioning guide<br/>Clear instructions]
        KK -->|System Error| NN[🔧 Technical retry<br/>Fallback processing]
        KK -->|Database Error| OO[💾 Storage retry<br/>Alternative DB path]
        
        LL --> C
        MM --> C
        NN --> G
        OO --> S
    end
    
    %% Connect error flows
    F -.-> JJ
    J1 -.-> JJ
    J2 -.-> JJ
    N1 -.-> JJ
    N2 -.-> JJ
    DD -.-> JJ

    %% Styling
    classDef input fill:#e3f2fd,stroke:#0277bd,stroke-width:2px,color:#000
    classDef process fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef ai1 fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef ai2 fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#000
    classDef db1 fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef db2 fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#000
    classDef validation fill:#f1f8e9,stroke:#689f38,stroke-width:2px,color:#000
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000
    
    class INPUT input
    class PROCESS process
    class AI1 ai1
    class AI2 ai2
    class DB1 db1
    class DB2 db2
    class VALIDATION,METRICS validation
    class ERROR error